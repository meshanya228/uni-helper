"""
/gossip текст сплетни

— Только в группах
— Сообщение удаляется немедленно (анонимность)
— Gemini перефразирует, чтобы стиль не выдал автора
— Попадает в ближайшую авто-сводку
"""

import logging

from telegram import Update
from telegram.ext import ContextTypes
from telegram.error import BadRequest
from telegram.constants import ChatType

from utils.db import save_gossip
from services.gemini import process_gossip

logger = logging.getLogger(__name__)


async def cmd_gossip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # Только группы
    if msg.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        # В личке — тихо игнорируем, не отвечаем
        return

    # Удаляем сразу
    try:
        await msg.delete()
    except BadRequest:
        pass

    raw = " ".join(context.args).strip() if context.args else ""
    if len(raw) < 5:
        # Тихо — не хотим выдавать что кто-то пытался написать сплетню
        return

    if len(raw) > 1000:
        raw = raw[:1000]

    try:
        processed = await process_gossip(raw)
        if not processed or len(processed.strip()) < 3:
            processed = raw  # фоллбэк
    except Exception:
        processed = raw

    await save_gossip(chat_id=msg.chat_id, gossip_text=processed.strip())
    logger.info(f"Gossip saved for chat {msg.chat_id}")
