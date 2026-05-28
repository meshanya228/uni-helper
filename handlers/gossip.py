"""
/gossip текст — анонимная сплетня.

— Только в группах
— Сообщение пользователя удаляется
— Бот отвечает коротко что сплетня принята (без деталей)
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
        return

    # Удаляем сообщение пользователя
    try:
        await msg.delete()
    except BadRequest as e:
        logger.warning(f"Can't delete gossip message: {e}")
        # Без прав удаления — сообщение останется видным, но всё равно сохраним
        # и уведомим что нужны права
        try:
            notice = await context.bot.send_message(
                msg.chat_id,
                "⚠️ Дай мне права на удаление сообщений для полной анонимности.")
            # Самоудаляющееся уведомление не делаем — просто логируем
        except Exception:
            pass

    raw = " ".join(context.args).strip() if context.args else ""
    if len(raw) < 3:
        # Тихо игнорируем пустые сплетни
        return

    if len(raw) > 1000:
        raw = raw[:1000]

    # Перефразируем через Gemini
    try:
        processed = await process_gossip(raw)
        if not processed or len(processed.strip()) < 3:
            processed = raw
    except Exception:
        processed = raw

    await save_gossip(chat_id=msg.chat_id, gossip_text=processed.strip())
    logger.info(f"Gossip saved for chat {msg.chat_id}")

    # Подтверждение — только боту видно (через отдельное сообщение которое сразу удаляем)
    # Или просто тихо, без подтверждения — анонимнее
    # Отправим в чат нейтральное сообщение
    try:
        confirm = await context.bot.send_message(
            msg.chat_id,
            "🤫 Сплетня принята. Всплывёт в следующей сводке.")
        # Удаляем подтверждение через 5 секунд
        import asyncio
        await asyncio.sleep(5)
        await confirm.delete()
    except Exception:
        pass
