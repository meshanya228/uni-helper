"""
/birthday ДД.ММ — сохранить или обновить день рождения.
- Если дата та же — сообщает что уже записана
- Если другая — обновляет
- Поздравление отправляется в Флудилку в 00:00 по Мадриду
- Дубликаты в БД исключены через PRIMARY KEY (user_id)
"""
import calendar
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import ContextTypes

from utils.db import (save_birthday, get_todays_birthdays,
                      update_birthday_profile, get_user_profile,
                      get_birthday_by_user)
from services.gemini import generate_birthday_message_bg

logger = logging.getLogger(__name__)
MADRID_TZ     = ZoneInfo("Europe/Madrid")
MAIN_TOPIC_ID = 1  # Флудилка


def _validate_date(raw: str):
    """Возвращает (day, month) или None."""
    try:
        parts = raw.strip().split(".")
        if len(parts) != 2:
            return None
        day, month = int(parts[0]), int(parts[1])
        if not (1 <= month <= 12):
            return None
        max_days = max(calendar.monthrange(2000, month)[1],
                       calendar.monthrange(2004, month)[1])
        if not (1 <= day <= max_days):
            return None
        return day, month
    except (ValueError, TypeError):
        return None


async def cmd_birthday(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    if not context.args:
        await msg.reply_text(
            "Формат: /birthday ДД.ММ\nНапример: /birthday 15.03")
        return

    result = _validate_date(context.args[0])
    if result is None:
        await msg.reply_text(
            "Неправильная дата. Нужно ДД.ММ, например 15.03 или 29.02")
        return

    day, month = result
    user = msg.from_user

    # Проверяем что уже есть в БД
    existing = await get_birthday_by_user(user.id)
    if existing:
        ex_day   = existing["birth_day"]
        ex_month = existing["birth_month"]
        if ex_day == day and ex_month == month:
            await msg.reply_text(
                f"Эта дата уже записана — {day:02d}.{month:02d} 🎂")
            return
        # Обновляем
        await save_birthday(
            user.id, user.username or "", user.first_name or "",
            day, month, msg.chat_id)
        await msg.reply_text(
            f"Обновил! Старая дата была {ex_day:02d}.{ex_month:02d}, "
            f"новая — {day:02d}.{month:02d} 🎂")
    else:
        await save_birthday(
            user.id, user.username or "", user.first_name or "",
            day, month, msg.chat_id)
        await msg.reply_text(
            f"Запомнил! Поздравлю тебя {day:02d}.{month:02d} 🎂")


async def check_birthdays(bot):
    """Вызывается планировщиком в 00:00 по Мадриду."""
    now  = datetime.now(MADRID_TZ)
    rows = await get_todays_birthdays(now.day, now.month)

    for row in rows:
        user_id    = row["user_id"]
        username   = row["username"] or "друг"
        first_name = row["first_name"] or username
        chat_id    = row["chat_id"]
        profile    = await get_user_profile(user_id)

        async def send_congrats(text, _cid=chat_id, _uid=user_id, _prof=profile):
            try:
                await bot.send_message(
                    chat_id=_cid, text=text,
                    parse_mode="Markdown",
                    message_thread_id=MAIN_TOPIC_ID)
            except Exception:
                try:
                    await bot.send_message(
                        chat_id=_cid, text=text, parse_mode="Markdown")
                except Exception as e:
                    logger.error(f"Birthday send failed: {e}")
            await update_birthday_profile(_uid, _prof)

        await generate_birthday_message_bg(first_name, username, profile, send_congrats)
