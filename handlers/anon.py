"""
/anon @user1 [@user2 ...] текст

— Только в группах
— Бот НЕМЕДЛЕННО удаляет сообщение пользователя (он как будто ничего не писал)
— Бот сам пишет: "📩 Анонимное сообщение для @user" + кнопка
— Кнопка показывает popup только адресатам
— Работает с несколькими адресатами
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
        await msg.reply_text("Эта команда работает только в группе.")
        return

    # Удаляем сообщение пользователя ПЕРВЫМ ДЕЛОМ
    try:
        await msg.delete()
    except BadRequest as e:
        logger.warning(f"Can't delete anon command message: {e}")
        # Нет прав на удаление — предупреждаем и выходим
        # (без прав анонимность не гарантирована)
        try:
            await context.bot.send_message(
                msg.chat_id,
                "⚠️ Дай мне права на удаление сообщений — иначе анонимность не работает.")
        except Exception:
            pass
        return

    raw = " ".join(context.args) if context.args else ""

    # Парсим юзернеймы
    usernames = _USERNAME_RE.findall(raw)
    if not usernames:
        try:
            await context.bot.send_message(
                msg.chat_id,
                "Укажи кому: /anon @username текст сообщения")
        except Exception:
            pass
        return

    # Текст — всё что идёт ПОСЛЕ последнего @mention
    last_pos = max(raw.rfind(f"@{u}") + len(f"@{u}") for u in usernames)
    text = raw[last_pos:].strip()

    if not text:
        try:
            await context.bot.send_message(
                msg.chat_id,
                "Напиши текст после юзернеймов: /anon @username твой текст")
        except Exception:
            pass
        return

    if len(text) > 1000:
        text = text[:1000]

    # Сохраняем в БД (все usernames в нижнем регистре)
    msg_id = await save_anon_message(
        chat_id=msg.chat_id,
        sender_id=msg.from_user.id,
        recipient_usernames=[u.lower() for u in usernames],
        message_text=text)

    recipients_display = " ".join(f"@{u}" for u in usernames)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📩 Прочитать", callback_data=f"anon:{msg_id}")
    ]])

    await context.bot.send_message(
        msg.chat_id,
        f"🔒 Анонимное сообщение для {recipients_display}",
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
        await query.answer(
            "Сообщение истекло (живёт 24 часа) или не существует.",
            show_alert=True)
        return

    recipients = [r.lower() for r in row["recipient_usernames"]]
    user_username = (user.username or "").lower().strip()

    # Проверяем доступ
    if not user_username or user_username not in recipients:
        await query.answer("Это сообщение не для тебя 😇", show_alert=False)
        return

    text = row["message_text"]
    # Telegram popup максимум 200 символов
    display = text if len(text) <= 200 else text[:197] + "…"
    await query.answer(display, show_alert=True)
