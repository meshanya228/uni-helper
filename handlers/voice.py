from telegram import Update
from telegram.ext import ContextTypes
from services.gemini import transcribe_audio_gemini


async def handle_voice_transcribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    target = msg.reply_to_message
    if not target or (not target.voice and not target.video_note):
        await msg.reply_text("Ответь этой командой на голосовое сообщение или кружочек 🎙")
        return

    fid   = target.voice.file_id if target.voice else target.video_note.file_id
    mtype = "audio/ogg" if target.voice else "video/mp4"

    f = await context.bot.get_file(fid)
    b = await f.download_as_bytearray()

    text = await transcribe_audio_gemini(bytes(b), mtype)
    await msg.reply_text(
        f"📝 *Расшифровка:*\n\n{text or 'Не удалось расшифровать'}",
        parse_mode="Markdown"
    )
