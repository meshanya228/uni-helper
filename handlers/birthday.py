import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from telegram import Update
from telegram.ext import ContextTypes
from utils.db import save_birthday, get_todays_birthdays, update_birthday_profile, get_user_profile
from services.gemini import generate_birthday_message_bg

logger = logging.getLogger(__name__)
MADRID_TZ = ZoneInfo("Europe/Madrid")
MAIN_TOPIC_ID = 1  # Флудилка


async def cmd_birthday(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not context.args:
        await msg.reply_text("Формат: `/birthday ДД.ММ` — например `/birthday 15.03`", parse_mode="Markdown")
        return

    raw = context.args[0].strip()
    try:
        day, month = map(int, raw.split("."))
        assert 1 <= day <= 31 and 1 <= month <= 12
    except Exception:
        await msg.reply_text("Неправильный формат. Нужно ДД.ММ, например `15.03`", parse_mode="Markdown")
        return

    user = msg.from_user
    await save_birthday(user.id, user.username or "", user.first_name or "", day, month, msg.chat_id)
    await msg.reply_text(f"Запомнил! Поздравлю тебя {day:02d}.{month:02d} 🎂")


async def check_birthdays(bot):
    """Запускается в полночь по Мадриду. Поздравляет именинников в Флудилке."""
    now = datetime.now(MADRID_TZ)
    rows = await get_todays_birthdays(now.day, now.month)

    for row in rows:
        user_id   = row["user_id"]
        username  = row["username"] or "друг"
        first_name = row["first_name"] or username
        chat_id   = row["chat_id"]
        profile   = await get_user_profile(user_id)

        async def send_congrats(text, cid=chat_id, uid=user_id):
            try:
                await bot.send_message(
                    chat_id=cid,
                    text=text,
                    parse_mode="Markdown",
                    message_thread_id=MAIN_TOPIC_ID
                )
            except Exception as e:
                logger.warning(f"Birthday thread send failed: {e}")
                try:
                    await bot.send_message(chat_id=cid, text=text, parse_mode="Markdown")
                except Exception as e2:
                    logger.error(f"Birthday send failed entirely: {e2}")
            await update_birthday_profile(uid, profile)

        await generate_birthday_message_bg(first_name, username, profile, send_congrats)
