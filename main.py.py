#!/usr/bin/python3

# الواردات القياسية
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

# الواردات الخارجية
import numpy as np
import librosa
import soundfile as sf
import psola
import asyncio
import concurrent.futures
import logging
from numba import jit

# واردات تيليجرام
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)

# إعداد السجلات
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()]
)

# التحقق من إصدار librosa
assert librosa.__version__ >= "0.10.0", "يجب استخدام إصدار 0.10.0 أو أحدث من librosa"

# الثوابت
SEMITONES_IN_OCTAVE = 12

ARABIC_MAQAMS = {
    "راست": {"degrees": [0, 2, 4.5, 5, 7, 9, 11], "description": "مقام أساسي في الموسيقى العربية، يتميز بالثبات والوضوح"},
    "بياتي": {"degrees": [0, 1.5, 3.5, 5, 7, 8.5, 10], "description": "من المقامات الأساسية، يعطي إحساساً بالحزن والأسى"},
    "صبا": {"degrees": [0, 1, 3.5, 5, 6.5, 8, 10], "description": "مقام الحزن والعاطفة العميقة"},
    "حجاز": {"degrees": [0, 1, 4, 5, 7, 8, 11], "description": "يستخدم في التعبير عن الشوق والحنين"},
    "سيكاه": {"degrees": [0.5, 1.5, 3.5, 5, 7, 9, 10.5], "description": "مقام العاطفة والرومانسية"},
    "نهاوند": {"degrees": [0, 2, 3, 5, 7, 8, 10], "description": "مقام الحزن العميق والكآبة"},
    "كرد": {"degrees": [0, 2, 3, 5, 7, 8, 10], "description": "يشبه النهاوند لكن بلمسة شرقية مميزة"},
    "عجم": {"degrees": [0, 2, 4, 5, 7, 9, 11], "description": "مقام الفرح والبهجة"},
    "جهاركاه": {"degrees": [0, 2.5, 3.5, 6, 7, 9, 10.5], "description": "مقام القوة والعظمة"},
}

# دوال المساعدة
@contextmanager
def temp_audio_file(suffix=".wav"):
    """إنشاء ملف مؤقت مع ضمان حذفه بعد الاستخدام."""
    temp_file = Path(tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name)
    try:
        yield temp_file
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception as e:
            logging.error(f"Error deleting temp file {temp_file}: {e}")

# معالجة الصوت
class AudioProcessor:
    @staticmethod
    def degrees_from(scale: str):
        degrees = librosa.key_to_degrees(scale)
        return np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))

    @staticmethod
    def closest_pitch(f0):
        midi_note = np.around(librosa.hz_to_midi(f0))
        midi_note[np.isnan(f0)] = np.nan
        return librosa.midi_to_hz(midi_note)

    @staticmethod
    def closest_pitch_from_scale(f0, scale_degrees, strength=1.0):
        if np.isnan(f0):
            return np.nan
        midi_note = librosa.hz_to_midi(f0)
        degree = midi_note % SEMITONES_IN_OCTAVE
        degree_id = np.argmin(np.abs(scale_degrees - degree))
        degree_difference = degree - scale_degrees[degree_id]
        midi_note -= degree_difference * strength
        return librosa.midi_to_hz(midi_note)

    @staticmethod
    def array_closest_pitch_from_scale(f0, scale, strength=1.0, progress_callback=None):
        scale_degrees = AudioProcessor.degrees_from(scale)
        sanitized_pitch = np.zeros_like(f0)
        total = f0.shape[0]
        for i in range(total):
            sanitized_pitch[i] = AudioProcessor.closest_pitch_from_scale(f0[i], scale_degrees, strength)
            if progress_callback and i % 1000 == 0:
                progress_callback(int((i / total) * 100))
        return sanitized_pitch

    @staticmethod
    @jit(nopython=True, cache=True)
    def closest_pitch_arabic(f0, maqam_freqs, strength=1.0):
        if np.isnan(f0):
            return np.nan
        diffs = np.abs(maqam_freqs - f0)
        closest_idx = np.argmin(diffs)
        closest_freq = maqam_freqs[closest_idx]
        return f0 * (1 - strength) + closest_freq * strength

    @staticmethod
    def array_closest_pitch_arabic(f0, maqam_freqs, strength=1.0, progress_callback=None):
        sanitized_pitch = np.zeros_like(f0)
        valid_mask = ~np.isnan(f0)
        diffs = np.abs(maqam_freqs[:, None] - f0[valid_mask])
        closest_indices = np.argmin(diffs, axis=0)
        closest_freqs = maqam_freqs[closest_indices]
        sanitized_pitch[valid_mask] = f0[valid_mask] * (1 - strength) + closest_freqs * strength
        sanitized_pitch[~valid_mask] = np.nan
        if progress_callback:
            progress_callback(100)
        return sanitized_pitch

    @staticmethod
    def autotune_chunk(audio, sr, correction_function, frame_length=1024, formant_correction=0.0, progress_callback=None):
        hop_length = frame_length // 4
        fmin = librosa.note_to_hz("C2")
        fmax = librosa.note_to_hz("C7")

        if progress_callback:
            progress_callback(10, "جاري كشف النغمات...")
            logging.debug("Autotune chunk: Pitch detection started")
            

        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                frame_length=frame_length,
                hop_length=hop_length,
                sr=sr,
                fmin=fmin,
                fmax=fmax,
                fill_na=np.nan,
                center=True,
                resolution=0.2
            )
        except Exception as e:
            logging.error(f"Error in pyin pitch detection: {str(e)}")
            raise

        if progress_callback:
            progress_callback(40, "جاري تصحيح النغمات...")
            logging.debug("Autotune chunk: Pitch correction started")
            

        corrected_f0 = correction_function(f0)

        if progress_callback:
            progress_callback(70, "جاري التعديل الصوتي...")
            logging.debug("Autotune chunk: Vocoding started")
            

        try:
            corrected_audio = psola.vocode(
                audio,
                sample_rate=int(sr),
                target_pitch=corrected_f0,
                fmin=fmin,
                fmax=fmax,
            )
        except Exception as e:
            logging.error(f"Error in psola.vocode: {str(e)}")
            raise

        if progress_callback:
            progress_callback(100, "اكتملت معالجة القطعة")
            logging.debug("Autotune chunk: Completed")

        return corrected_audio

    @staticmethod
    def autotune(
        audio,
        sr,
        correction_function,
        frame_length=1024,
        formant_correction=0.0,
        progress_callback=None,
        chunk_size=44100 * 2
    ):
        logging.debug(f"Starting autotune: audio length={len(audio)}, sample_rate={sr}")
        total_samples = len(audio)
        processed_audio = np.zeros_like(audio, dtype=np.float32)
        chunk_progress = 0
        chunk_count = (total_samples + chunk_size - 1) // chunk_size
        current_chunk = 0

        if progress_callback:
            progress_callback(0, "بدء معالجة القطع...")
            logging.debug("Autotune: Starting chunk processing")

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            current_chunk += 1
            logging.debug(f"Processing chunk: start={start}, end={end}")
            try:
                chunk = audio[start:end]
                chunk_processed = AudioProcessor.autotune_chunk(
                    chunk,
                    sr,
                    correction_function,
                    frame_length,
                    formant_correction,
                    lambda p, stage: progress_callback(p * (end - start) / total_samples / chunk_count + chunk_progress, f"معالجة القطعة {current_chunk}/{chunk_count}: {stage}")
                )
                processed_audio[start:end] = chunk_processed
                chunk_progress += 100 * (end - start) / total_samples
                if progress_callback:
                    progress_callback(chunk_progress, f"اكتملت القطعة {current_chunk}/{chunk_count}")
                    logging.debug(f"Chunk {current_chunk}/{chunk_count} completed: {chunk_progress:.1f}%")
            except Exception as e:
                logging.error(f"Error processing chunk {start}-{end}: {str(e)}")
                raise
        logging.debug("Autotune completed")
        return processed_audio

    @staticmethod
    def get_maqam_frequencies(root_note, maqam_name, octaves=2):
        base_freq = librosa.note_to_hz(root_note + "3")
        degrees = ARABIC_MAQAMS[maqam_name]["degrees"]
        freqs = np.array([base_freq * (2 ** (degree / 12)) * (2 ** octave) for octave in range(octaves) for degree in degrees], dtype=np.float32)
        return freqs

# حالات المحادثة
(
    UPLOAD_AUDIO,
    SELECT_SCALE_TYPE,
    SELECT_WESTERN_SCALE,
    SELECT_ARABIC_MAQAM,
    SET_STRENGTH,
    PROCESS_AUDIO,
) = range(6)

# دوال المحادثة
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """بدء المحادثة وطلب تحميل ملف صوتي."""
    await update.message.reply_text(
        "مرحبًا! أنا بوت طورني صبحي شبايكي بالإشراف وتوجيه أدوات الذكاء الإصطناعي لمعالجة الصوت الموسيقي. "
        "تابعني على فيسبوك: https://www.facebook.com/chebaiki.sobhi.official\n"
        "يرجى إرسال ملف صوتي (wav، mp3، أو ogg) لمعالجته."
    )
    return UPLOAD_AUDIO

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """التعامل مع الملف الصوتي المرفوع."""
    valid_extensions = {'.wav', '.mp3', '.ogg'}
    audio = update.message.audio or update.message.document
    if not audio:
        await update.message.reply_text("يرجى إرسال ملف صوتي (wav، mp3، أو ogg).")
        return UPLOAD_AUDIO

    file_ext = os.path.splitext(audio.file_name)[1].lower()
    if file_ext not in valid_extensions:
        await update.message.reply_text("تنسيق الملف غير مدعوم. استخدم wav، mp3، أو ogg.")
        return UPLOAD_AUDIO

    try:
        temp_file = Path(tempfile.NamedTemporaryFile(suffix=file_ext, delete=False).name)
        file = await audio.get_file()
        await file.download_to_drive(temp_file)
        audio_data, sr = librosa.load(temp_file, sr=None, mono=True, duration=10)
        duration = librosa.get_duration(path=temp_file, sr=sr)
        if duration > 300:
            await update.message.reply_text("الملف طويل جدًا. الحد الأقصى هو 5 دقائق.")
            temp_file.unlink(missing_ok=True)
            return UPLOAD_AUDIO

        # التحقق من حجم الملف
        file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
        if file_size_mb > 50:
            await update.message.reply_text("الملف كبير جدًا (أكثر من 50 ميجابايت). يرجى استخدام ملف أصغر.")
            temp_file.unlink(missing_ok=True)
            return UPLOAD_AUDIO

        context.user_data['audio_path'] = str(temp_file)
        context.user_data['sample_rate'] = sr
        context.user_data['temp_file'] = temp_file

        keyboard = [
            [InlineKeyboardButton("غربي", callback_data="western")],
            [InlineKeyboardButton("شرقي", callback_data="arabic")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            f"تم تحميل الملف: {audio.file_name} ({duration:.2f} ث)\nاختر نوع السلم الموسيقي:",
            reply_markup=reply_markup
        )
        return SELECT_SCALE_TYPE

    except Exception as e:
        await update.message.reply_text(f"فشل تحميل الملف: {str(e)}")
        logging.error(f"Error loading audio: {str(e)}")
        if 'temp_file' in locals():
            temp_file.unlink(missing_ok=True)
        return UPLOAD_AUDIO

async def select_scale_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """اختيار نوع السلم (غربي أو شرقي)."""
    query = update.callback_query
    await query.answer()
    scale_type = query.data

    context.user_data['scale_type'] = scale_type

    if scale_type == "western":
        keyboard = []
        for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            keyboard.append([
                InlineKeyboardButton(f"{note}:major", callback_data=f"{note}:major"),
                InlineKeyboardButton(f"{note}:minor", callback_data=f"{note}:minor")
            ])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("اختر الجذر والنمط للسلم الغربي:", reply_markup=reply_markup)
        return SELECT_WESTERN_SCALE
    else:
        keyboard = [[InlineKeyboardButton(maqam, callback_data=maqam)] for maqam in ARABIC_MAQAMS.keys()]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("اختر المقام العربي:", reply_markup=reply_markup)
        return SELECT_ARABIC_MAQAM

async def select_western_scale(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """اختيار السلم الغربي (الجذر والنمط)."""
    query = update.callback_query
    await query.answer()
    root, mode = query.data.split(':')
    context.user_data['root_note'] = root
    context.user_data['mode'] = mode

    keyboard = [
        [InlineKeyboardButton(f"{int(p*100)}%", callback_data=str(p)) for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text("اختر قوة الضبط (نسبة التأثير):", reply_markup=reply_markup)
    return SET_STRENGTH

async def select_arabic_maqam(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """اختيار المقام العربي."""
    query = update.callback_query
    await query.answer()
    maqam = query.data
    context.user_data['maqam'] = maqam

    await query.message.reply_text(f"وصف المقام {maqam}: {ARABIC_MAQAMS[maqam]['description']}")

    keyboard = [
        [InlineKeyboardButton(f"{int(p*100)}%", callback_data=str(p)) for p in [0.0, 0.25, 0.5, 0.75, 1.0]]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text("اختر قوة الضبط (نسبة التأثير):", reply_markup=reply_markup)
    return SET_STRENGTH

async def set_strength(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """تحديد قوة الضبط."""
    query = update.callback_query
    await query.answer()
    strength = float(query.data)
    context.user_data['strength'] = strength

    keyboard = [
        [InlineKeyboardButton("ابدأ المعالجة", callback_data="process")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.message.reply_text(
        f"تم تحديد قوة الضبط: {int(strength*100)}%\nاضغط للبدء في المعالجة:",
        reply_markup=reply_markup
    )
    return PROCESS_AUDIO

async def process_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """معالجة الملف الصوتي مع إرسال رسائل لكل مرحلة."""
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("جاري التهيئة ...")

    if query.data != "process":
        await query.message.reply_text("عملية غير صالحة. يرجى البدء من جديد باستخدام /start.")
        return ConversationHandler.END

    audio_path = context.user_data.get('audio_path')
    if not audio_path:
        await query.message.reply_text("لم يتم تحميل ملف صوتي. يرجى البدء من جديد باستخدام /start.")
        return ConversationHandler.END

    logging.debug("Starting audio processing")

    # إرسال رسالة المرحلة الأولية
    try:
        await query.message.reply_text("جاري تحميل الصوت...")
        logging.debug("Initial stage message sent")
    except Exception as e:
        logging.error(f"Error sending initial stage message: {str(e)}")
        await query.message.reply_text("خطأ في بدء المعالجة. حاول مرة أخرى.")
        return ConversationHandler.END

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        try:
            logging.debug(f"Loading audio file: {audio_path}")
            audio, sr = await loop.run_in_executor(None, lambda: librosa.load(audio_path, sr=context.user_data['sample_rate'], mono=True, dtype=np.float32))
            strength = context.user_data['strength']
            formant_correction = 0.0
            frame_length = 1024

            if context.user_data['scale_type'] == "arabic":
                root = "C"
                maqam = context.user_data['maqam']
                logging.debug(f"Using Arabic maqam: {maqam}")
                maqam_freqs = AudioProcessor.get_maqam_frequencies(root, maqam)
                correction_function = lambda x: AudioProcessor.array_closest_pitch_arabic(x, maqam_freqs, strength)
            else:
                root = context.user_data['root_note']
                mode = context.user_data['mode']
                scale = f"{root}:{mode}"
                logging.debug(f"Using Western scale: {scale}")
                correction_function = lambda x: AudioProcessor.array_closest_pitch_from_scale(x, scale, strength)

            async def progress_callback(progress, stage):
                try:
                    await query.message.reply_text(stage)
                    logging.debug(f"Stage message sent: {stage}")
                except Exception as e:
                    logging.error(f"Error sending stage message: {str(e)}")

            logging.debug("Starting autotune processing")
            await query.message.reply_text("بدء معالجة الضبط التلقائي قد يستغرق وقت حسب مدة الصوت ...")
            processed_audio = await asyncio.wait_for(
                loop.run_in_executor(
                    pool,
                    lambda: AudioProcessor.autotune(
                        audio,
                        sr,
                        correction_function,
                        frame_length,
                        formant_correction,
                        progress_callback
                    )
                ),
                timeout=300
            )

            logging.debug("Writing processed audio to file")
            await query.message.reply_text("كتابة الصوت المعالج إلى ملف")
            with temp_audio_file(suffix=".wav") as output_file:
                await loop.run_in_executor(None, lambda: sf.write(output_file, processed_audio, sr))
                
                # التحقق من حجم الملف الناتج
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                if file_size_mb > 50:
                    await query.message.reply_text("الملف الناتج كبير جدًا (أكثر من 50 ميجابايت). يرجى تجربة ملف أصغر.")
                    return ConversationHandler.END

                logging.debug(f"Sending processed audio: {output_file}")
                with open(output_file, 'rb') as f:
                    await query.message.reply_audio(audio=f, title="processed_audio.wav")

            await query.message.reply_text("تمت المعالجة بنجاح!")
            logging.debug("Audio processing completed successfully")
            return ConversationHandler.END

        except asyncio.TimeoutError:
            await query.message.reply_text("تجاوزت المعالجة الوقت المحدد (5 دقائق). حاول مع ملف أصغر.")
            logging.error("Processing timed out after 5 minutes")
            return ConversationHandler.END
        except librosa.LibrosaError as e:
            await query.message.reply_text("خطأ في معالجة الصوت: تنسيق الملف غير مدعوم أو تالف.")
            logging.error(f"Librosa error: {str(e)}")
            return ConversationHandler.END
        except Exception as e:
            await query.message.reply_text(f"حدث خطأ أثناء المعالجة: {str(e)}")
            logging.error(f"Processing error: {str(e)}")
            return ConversationHandler.END
        finally:
            if 'temp_file' in context.user_data:
                context.user_data['temp_file'].unlink(missing_ok=True)
            context.user_data.clear()
            logging.debug("Cleaned up user data and temporary files")

# دوال إضافية
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """إلغاء العملية."""
    await update.message.reply_text("تم إلغاء العملية. ابدأ من جديد باستخدام /start.")
    if 'temp_file' in context.user_data:
        context.user_data['temp_file'].unlink(missing_ok=True)
    context.user_data.clear()
    logging.debug("Operation cancelled by user")
    return ConversationHandler.END

def main() -> None:
    """تشغيل البوت."""
    application = Application.builder().token("8050283100:AAF07B8t__h3ffyIANkn0CBnQJM6DLrlr54").build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            UPLOAD_AUDIO: [MessageHandler(filters.AUDIO | filters.Document.ALL, handle_audio)],
            SELECT_SCALE_TYPE: [CallbackQueryHandler(select_scale_type)],
            SELECT_WESTERN_SCALE: [CallbackQueryHandler(select_western_scale)],
            SELECT_ARABIC_MAQAM: [CallbackQueryHandler(select_arabic_maqam)],
            SET_STRENGTH: [CallbackQueryHandler(set_strength)],
            PROCESS_AUDIO: [CallbackQueryHandler(process_audio)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    main()
