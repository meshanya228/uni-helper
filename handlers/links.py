import logging
from telegram import Update
from telegram.ext import ContextTypes
from utils.db import add_link, get_all_links, delete_link, get_link_by_id
from services.gemini import validate_and_format_link

logger = logging.getLogger(__name__)


async def cmd_links(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    if not context.args:
        await _show_links(msg, msg.chat_id)
        return
    subcmd = context.args[0].lower()
    if subcmd == "add":
        await _add_link(msg, msg.chat_id, " ".join(context.args[1:]))
    elif subcmd in ("del", "delete") and len(context.args) > 1:
        try:
            await _delete_link(msg, msg.chat_id, int(context.args[1]))
        except ValueError:
            await msg.reply_text("Укажи числовой ID: `/links del 5`", parse_mode="Markdown")
    else:
        await msg.reply_text("Команды: `/links`, `/links add [ссылка]`, `/links del [ID]`", parse_mode="Markdown")


async def _show_links(msg, chat_id):
    rows = await get_all_links(chat_id)
    if not rows:
        await msg.reply_text("Список ссылок пуст. Добавь первую: `/links add ссылка описание`", parse_mode="Markdown")
        return
    lines = ["📎 *Полезные ссылки:*\n"]
    current_tag = None
    for r in rows:
        if r["tag"] != current_tag:
            current_tag = r["tag"]
            lines.append(f"\n*{current_tag}*")
        desc = f" — {r['description']}" if r["description"] else ""
        lines.append(f"  #{r['id']} [{r['title']}]({r['url']}){desc}")
    await msg.reply_text("\n".join(lines), parse_mode="Markdown", disable_web_page_preview=True)


async def _add_link(msg, chat_id, raw):
    if not raw:
        await msg.reply_text("Что добавить? `/links add https://... описание`", parse_mode="Markdown")
        return
    res = await validate_and_format_link(raw)
    if not res.get("ok"):
        await msg.reply_text(f"❌ Не добавил: {res.get('reason', 'что-то пошло не так')}")
        return
    lid = await add_link(
        chat_id, res["title"], res["url"], res["tag"],
        res["description"], msg.from_user.id, msg.from_user.first_name
    )
    await msg.reply_text(f"✅ Добавлено #{lid}: *{res['title']}* ({res['tag']})", parse_mode="Markdown")


async def _delete_link(msg, chat_id, link_id):
    row = await get_link_by_id(chat_id, link_id)
    if not row:
        await msg.reply_text("Ссылка не найдена.")
        return
    # Удалить может тот кто добавил, или если нет данных
    if row["added_by_id"] and row["added_by_id"] != msg.from_user.id:
        # Проверяем — мягко, не блокируем (для простоты без прав адм)
        pass
    await delete_link(chat_id, link_id)
    await msg.reply_text(f"🗑 Ссылка #{link_id} удалена.")
