"""
/anon @user1 @user2 текст сообщения

- Исходное сообщение бот удаляет (или просит удалить если нет прав)
- В чат летит сообщение с кнопкой «📩 Тебе сообщение»
- По кнопке — popup (answerCallbackQuery) с текстом, видим только адресатам
- Сообщение в БД живёт 24 часа
"""

import logging
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest
from utils.db import save_anon_message, get_anon_message

logger = logging.getLogger(__name__)

_USERNAME_RE = re.compile(r"@([\w]+)")


async def cmd_anon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # Удаляем исходное сообщение сразу — оно не должно быть видно в чате
    try:
        await msg.delete()
    except BadRequest:
        pass  # нет прав — просто продолжаем, сообщение останется

    raw = " ".join(context.args) if context.args else ""
    if not raw:
        try:
            await context.bot.send_message(
                msg.chat_id,
                "Использование: `/anon @user1 [@user2 ...] текст`",
                parse_mode="Markdown"
            )
        except Exception:
            pass
        return

    # Парсим получателей и текст
    usernames = _USERNAME_RE.findall(raw)
    if not usernames:
        try:
            await context.bot.send_message(
                msg.chat_id,
                "Укажи хотя бы одного получателя: `/anon @username текст`",
                parse_mode="Markdown"
            )
        except Exception:
            pass
        return

    # Текст — всё что после последнего @username
    last_mention = max(raw.rfind(f"@{u}") + len(f"@{u}") for u in usernames)
    text = raw[last_mention:].strip()
    if not text:
        try:
            await context.bot.send_message(
                msg.chat_id,
                "А текст сообщения где? `/anon @username твой текст`",
                parse_mode="Markdown"
            )
        except Exception:
            pass
        return

    # Сохраняем в БД
    msg_id = await save_anon_message(
        chat_id=msg.chat_id,
        sender_id=msg.from_user.id,
        recipient_usernames=[u.lower() for u in usernames],
        message_text=text
    )

    # Формируем список получателей для отображения
    recipients_display = " ".join(f"@{u}" for u in usernames)

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📩 Тебе сообщение", callback_data=f"anon:{msg_id}")
    ]])

    await context.bot.send_message(
        msg.chat_id,
        f"🔒 *Анонимное сообщение* для {recipients_display}",
        parse_mode="Markdown",
        reply_markup=keyboard
    )


async def handle_anon_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return

    user = query.from_user
    _, msg_id_str = query.data.split(":", 1)
    msg_id = int(msg_id_str)

    row = await get_anon_message(msg_id)

    if row is None:
        await query.answer("Сообщение истекло или не существует 🕳️", show_alert=True)
        return

    recipients = [r.lower() for r in row["recipient_usernames"]]
    user_username = (user.username or "").lower()

    if user_username not in recipients:
        await query.answer("Это не тебе 😇", show_alert=False)
        return

    # Показываем текст в popup — только адресату видно
    text = row["message_text"]
    # Telegram ограничивает popup до 200 символов
    display = text if len(text) <= 200 else text[:197] + "..."
    await query.answer(display, show_alert=True)
