from telegram import Update
from telegram.ext import ContextTypes

MAIN_TOPIC_ID = 1  # Флудилка


async def handle_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.chat_member or not update.chat_member.new_chat_member:
        return
    u = update.chat_member.new_chat_member.user
    if u.is_bot:
        return

    chat_id = update.chat_member.chat.id
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"👋 Привет, {u.mention_html()}! Добро пожаловать в UAшники!\n\nРасскажи, на кого учишься и откуда приехал?",
            parse_mode="HTML",
            message_thread_id=MAIN_TOPIC_ID
        )
    except Exception:
        # Если тема не работает — без thread
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"👋 Привет, {u.mention_html()}! Добро пожаловать в UAшники!\n\nРасскажи, на кого учишься и откуда приехал?",
            parse_mode="HTML"
        )
