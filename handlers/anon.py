"""
/anon @user1 [@user2 ...] текст

— Только в групповых чатах
— Исходное сообщение удаляется
— В чат летит кнопка «📩 Тебе сообщение»
— Popup виден только адресатам (по username)
— Живёт 24 часа
"""

import logging
import re

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.error import BadRequest
from telegram.constants import ChatType

from utils.db import save_anon_message, get_anon_message

logger = logging.getLogger(__name__)
_USERNAME_RE = re.compile(r"@([\w]+)")


async def cmd_anon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # Только группы
    if msg.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Эта команда работает только в групповом чате.")
        return

    # Удаляем сразу — анонимность важнее
    try:
        await msg.delete()
    except BadRequest:
        pass

    raw = " ".join(context.args) if context.args else ""

    usernames = _USERNAME_RE.findall(raw)
    if not usernames:
        try:
            await context.bot.send_message(
                msg.chat_id,
                "Укажи получателя: `/anon @username текст сообщения`",
                parse_mode="Markdown")
        except Exception:
            pass
        return

    # Текст — всё после последнего упомянутого @username
    last_pos = max(raw.rfind(f"@{u}") + len(f"@{u}") for u in usernames)
    text = raw[last_pos:].strip()
    if not text:
        try:
            await context.bot.send_message(
                msg.chat_id,
                "Напиши текст сообщения после юзернеймов.",
                parse_mode="Markdown")
        except Exception:
            pass
        return

    if len(text) > 1000:
        text = text[:1000]

    msg_id = await save_anon_message(
        chat_id=msg.chat_id,
        sender_id=msg.from_user.id,
        recipient_usernames=[u.lower() for u in usernames],
        message_text=text)

    recipients_display = " ".join(f"@{u}" for u in usernames)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📩 Тебе сообщение", callback_data=f"anon:{msg_id}")
    ]])

    await context.bot.send_message(
        msg.chat_id,
        f"🔒 *Анонимное сообщение* для {recipients_display}",
        parse_mode="Markdown",
        reply_markup=keyboard)


async def handle_anon_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return

    user = query.from_user
    try:
        msg_id = int(query.data.split(":", 1)[1])
    except (IndexError, ValueError):
        await query.answer("Что-то пошло не так.", show_alert=False)
        return

    row = await get_anon_message(msg_id)
    if row is None:
        await query.answer("Сообщение истекло или не существует 🕳️", show_alert=True)
        return

    recipients = [r.lower() for r in row["recipient_usernames"]]
    # username может быть None — тогда точно не адресат
    user_username = (user.username or "").lower()

    if not user_username or user_username not in recipients:
        await query.answer("Это не тебе 😇", show_alert=False)
        return

    text = row["message_text"]
    # Telegram popup: максимум 200 символов
    display = text if len(text) <= 200 else text[:197] + "…"
    await query.answer(display, show_alert=True)
