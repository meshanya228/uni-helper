import random
import logging
from telegram import Update
from telegram.ext import ContextTypes
from utils.db import log_message, upsert_user_profile
from services.gemini import ask_gemini_background

logger = logging.getLogger(__name__)


def _parse_message(msg) -> tuple[str, str]:
    """Определить тип и контент сообщения. Возвращает (msg_type, content)."""
    if msg.text:
        return "text", msg.text
    elif msg.voice:
        return "voice", ""
    elif msg.video_note:
        return "video_note", ""
    elif msg.photo:
        # Сохраняем подпись если есть
        return "photo", msg.caption or ""
    elif msg.video:
        return "video", msg.caption or ""
    elif msg.animation:
        return "animation", msg.caption or ""
    elif msg.sticker:
        emoji = msg.sticker.emoji or ""
        return "sticker", emoji
    elif msg.document:
        name = msg.document.file_name or ""
        return "document", name
    elif msg.audio:
        return "audio", msg.caption or ""
    elif msg.location:
        return "location", ""
    elif msg.poll:
        return "poll", msg.poll.question or ""
    else:
        return "other", ""


async def handle_any_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.from_user or msg.from_user.is_bot:
        return

    user = msg.from_user
    mtype, content = _parse_message(msg)

    # Логируем всё
    await log_message(
        chat_id=msg.chat_id,
        user_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        msg_type=mtype,
        content=content
    )

    # Профиль — обновляем всегда, заметки генерируем с вероятностью 10% для текста
    if mtype == "text" and len(content) > 20 and random.random() < 0.1:
        async def save_prof(notes):
            await upsert_user_profile(
                user.id, msg.chat_id, user.username or "",
                user.first_name or "", notes
            )
        await ask_gemini_background(
            f"Одной короткой фразой (макс 30 слов) — что можно сказать о личности человека по этому сообщению: «{content}». Только заметка, без вступлений.",
            save_prof,
            use_search=False
        )
    else:
        await upsert_user_profile(
            user.id, msg.chat_id, user.username or "",
            user.first_name or "", ""
        )
