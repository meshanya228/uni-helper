import logging

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatType

from utils.db import (
    get_yesterday_messages, check_summary_used, mark_summary_used,
    cleanup_old_messages, get_user_by_username, get_user_messages,
    get_user_profile, get_yesterdays_gossips, get_auto_summary, toggle_auto_summary)
from services.gemini import ask_gemini_interactive, generate_summary, generate_user_profile

logger = logging.getLogger(__name__)
MAIN_TOPIC_ID = 1  # Флудилка


def _build_log_lines(rows) -> str:
    lines = []
    for r in rows:
        mtype   = r["msg_type"]
        name    = r["first_name"] or r["username"] or "кто-то"
        content = r["content"] or ""
        if mtype == "text":
            lines.append(f"{name}: {content}")
        elif mtype == "voice":
            lines.append(f"{name}: [голосовое]")
        elif mtype == "video_note":
            lines.append(f"{name}: [кружочек]")
        elif mtype == "photo":
            lines.append(f"{name}: [фото{': ' + content if content else ''}]")
        elif mtype == "video":
            lines.append(f"{name}: [видео{': ' + content if content else ''}]")
        elif mtype == "sticker":
            lines.append(f"{name}: [стикер]")
        elif mtype == "animation":
            lines.append(f"{name}: [гифка]")
        elif mtype == "poll":
            lines.append(f"{name}: [опрос: {content}]")
        else:
            lines.append(f"{name}: [{mtype}]")
    return "\n".join(lines)


async def cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    if not context.args:
        await msg.reply_text("Напиши вопрос после /ai")
        return
    prompt = " ".join(context.args)
    # Включаем поиск только если вопрос про что-то актуальное/текущее
    search_keywords = ("сейчас", "сегодня", "вчера", "2024", "2025", "новост",
                       "курс", "погод", "расписани", "когда", "кто такой",
                       "что случил", "последн", "актуальн", "недавно")
    use_search = any(kw in prompt.lower() for kw in search_keywords)
    res = await ask_gemini_interactive(prompt, use_search=use_search)
    await msg.reply_text(res)


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ручной вызов сводки — работает всегда."""
    msg = update.message
    if not msg:
        return

    if msg.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Сводка только в групповом чате.")
        return

    chat_id = msg.chat_id
    if await check_summary_used(chat_id):
        await msg.reply_text("Сводка на сегодня уже была. Приходи завтра.")
        return

    rows = await get_yesterday_messages(chat_id)
    if not rows:
        await msg.reply_text("Вчера в чате было тихо — нечего пересказывать.")
        return

    text = _build_log_lines(rows)
    gossip_rows = await get_yesterdays_gossips(chat_id)
    gossips = [g["gossip_text"] for g in gossip_rows] if gossip_rows else None

    summary = await generate_summary(text, gossips=gossips)
    await mark_summary_used(chat_id)
    await cleanup_old_messages(chat_id)
    await msg.reply_text(f"📊 *Сводка за вчера:*\n\n{summary}", parse_mode="Markdown")


async def cmd_summary_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    if msg.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Эта команда работает только в группе.")
        return

    result = await toggle_auto_summary(msg.chat_id)
    if result is None:
        await msg.reply_text("Слишком часто переключаешь — попробуй завтра.")
    elif result:
        await msg.reply_text("✅ Авто-сводка включена. Каждый день в 09:00 по Мадриду.")
    else:
        await msg.reply_text("🔕 Авто-сводка выключена. Вызвать вручную: /summary")


async def cmd_who(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/who @username — характеристика пользователя по его сообщениям."""
    msg = update.message
    if not msg:
        return
    if msg.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await msg.reply_text("Эта команда работает только в группе.")
        return
    if not context.args:
        await msg.reply_text("Укажи пользователя: `/who @username`", parse_mode="Markdown")
        return

    raw_username = context.args[0].lstrip("@").lower()
    if not raw_username:
        await msg.reply_text("Нет юзернейма.")
        return

    chat_id = msg.chat_id
    row = await get_user_by_username(chat_id, raw_username)
    if not row:
        await msg.reply_text(
            f"Не знаю @{raw_username} — может, никогда не писал в чате?")
        return

    user_id    = row["user_id"]
    first_name = row["first_name"] or raw_username
    username   = row["username"] or raw_username

    messages      = await get_user_messages(chat_id, user_id, limit=200)
    profile_notes = await get_user_profile(user_id)

    if not messages and not profile_notes:
        await msg.reply_text(f"По @{username} данных почти нет — тёмная лошадка.")
        return

    wait_msg = await msg.reply_text("анализирую...")
    result = await generate_user_profile(username, first_name, messages, profile_notes)

    try:
        await wait_msg.delete()
    except Exception:
        pass

    await msg.reply_text(result)


async def send_auto_summary(bot, chat_id: int):
    """Вызывается планировщиком. Шлёт сводку в Флудилку."""
    if not await get_auto_summary(chat_id):
        return
    if await check_summary_used(chat_id):
        return

    rows = await get_yesterday_messages(chat_id)
    if not rows:
        return

    text = _build_log_lines(rows)
    gossip_rows = await get_yesterdays_gossips(chat_id)
    gossips = [g["gossip_text"] for g in gossip_rows] if gossip_rows else None

    summary = await generate_summary(text, gossips=gossips)
    await mark_summary_used(chat_id)
    await cleanup_old_messages(chat_id)

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=f"📊 *Сводка за вчера:*\n\n{summary}",
            parse_mode="Markdown",
            message_thread_id=MAIN_TOPIC_ID)
    except Exception:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=f"📊 *Сводка за вчера:*\n\n{summary}",
                parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Auto summary send failed: {e}")
