"""
/gossip текст сплетни

- Сообщение пользователя удаляется из чата
- Сплетня сохраняется в БД анонимно
- При генерации summary вчерашние сплетни вплетаются в текст
"""

import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.error import BadRequest
from utils.db import save_gossip
from services.gemini import process_gossip

logger = logging.getLogger(__name__)


async def cmd_gossip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # Удаляем сообщение немедленно — анонимность прежде всего
    try:
        await msg.delete()
    except BadRequest:
        pass

    raw = " ".join(context.args).strip() if context.args else ""
    if not raw or len(raw) < 5:
        # Тихо не отвечаем — чтобы не было видно что человек пытался отправить сплетню
        return

    if len(raw) > 1000:
        raw = raw[:1000]

    # Перефразируем через Gemini чтобы стиль автора не выдал его
    try:
        processed = await process_gossip(raw)
    except Exception:
        processed = raw  # фоллбэк — сохраняем как есть

    await save_gossip(chat_id=msg.chat_id, gossip_text=processed)
    logger.info(f"Gossip saved for chat {msg.chat_id}")
