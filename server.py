import asyncio
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from aiohttp import web, ClientSession
from telegram import Update, BotCommand
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ChatMemberHandler, CallbackQueryHandler
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from handlers.voice    import handle_voice_transcribe
from handlers.birthday import cmd_birthday, check_birthdays
from handlers.welcome  import handle_new_member
from handlers.ai       import cmd_ai, cmd_summary, cmd_summary_toggle, cmd_who, send_auto_summary
from handlers.collect  import handle_any_message
from handlers.links    import cmd_links
from handlers.anon     import cmd_anon, handle_anon_callback
from handlers.gossip   import cmd_gossip
from handlers.map_cmd  import cmd_map
from utils.db          import init_db, cleanup_expired_anon_messages, cleanup_old_gossips
from services.gemini   import ensure_bg_worker

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
MADRID_TZ = ZoneInfo("Europe/Madrid")

# ID основного чата для авто-сводки (Флудилка)
# Если бот работает в нескольких чатах — можно расширить логику
MAIN_CHAT_ID = int(os.environ.get("MAIN_CHAT_ID", "0"))


# ─── HTTP keepalive ────────────────────────────────────────────────────────────

async def health(request):
    return web.Response(text="UniHelper is alive 🤖", status=200)

async def start_http_server():
    app = web.Application()
    app.router.add_get("/",       health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 7860)
    await site.start()
    logger.info("HTTP keepalive server started on :7860")


# ─── Self-pinger (Render не засыпает) ─────────────────────────────────────────

async def self_ping():
    """Пингует себя каждые 10 минут. Окно тишины 04:00–06:00 по Мадриду."""
    now = datetime.now(MADRID_TZ)
    if 4 <= now.hour < 6:
        return
    url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:7860") + "/health"
    try:
        async with ClientSession() as s:
            async with s.get(url, timeout=10) as r:
                logger.debug(f"Self-ping: {r.status}")
    except Exception as e:
        logger.warning(f"Self-ping failed: {e}")


# ─── /start handler ────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context):
    text = (
        "Привет! Я UniHelper — бот UAшников 🎓\n\n"
        "/voice — расшифровать голосовое или кружочек\n"
        "/birthday ДД.ММ — сохранить день рождения\n"
        "/ai [вопрос] — спросить у ИИ\n"
        "/summary — сводка за вчера\n"
        "/summarytoggle — вкл/выкл авто-сводку\n"
        "/who @user — характеристика пользователя\n"
        "/anon @user текст — анонимное сообщение\n"
        "/gossip текст — анонимная сплетня (попадёт в сводку)\n"
        "/links — полезные ссылки\n"
        "/map — карта СССР в Аликанте"
    )
    await update.message.reply_text(text)


# ─── Application builder ──────────────────────────────────────────────────────

async def build_application() -> Application:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app   = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start",         cmd_start))
    app.add_handler(CommandHandler("help",          cmd_start))
    app.add_handler(CommandHandler("voice",         handle_voice_transcribe))
    app.add_handler(CommandHandler("birthday",      cmd_birthday))
    app.add_handler(CommandHandler("ai",            cmd_ai))
    app.add_handler(CommandHandler("summary",       cmd_summary))
    app.add_handler(CommandHandler("summarytoggle", cmd_summary_toggle))
    app.add_handler(CommandHandler("who",           cmd_who))
    app.add_handler(CommandHandler("links",         cmd_links))
    app.add_handler(CommandHandler("anon",          cmd_anon))
    app.add_handler(CommandHandler("gossip",        cmd_gossip))
    app.add_handler(CommandHandler("map",           cmd_map))

    # Inline кнопки для анонимных сообщений
    app.add_handler(CallbackQueryHandler(handle_anon_callback, pattern=r"^anon:\d+$"))

    # Новые участники
    app.add_handler(ChatMemberHandler(handle_new_member, ChatMemberHandler.CHAT_MEMBER))

    # Все остальные сообщения — логируем
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_any_message))

    return app


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    await start_http_server()
    await init_db()
    ensure_bg_worker()

    app = await build_application()

    await app.bot.set_my_commands([
        BotCommand("start",         "Информация о боте"),
        BotCommand("help",          "Справка по командам"),
        BotCommand("voice",         "Расшифровать голосовое / кружочек"),
        BotCommand("birthday",      "Сохранить день рождения (ДД.ММ)"),
        BotCommand("ai",            "Спросить у ИИ"),
        BotCommand("summary",       "Сводка за вчера"),
        BotCommand("summarytoggle", "Вкл/выкл авто-сводку"),
        BotCommand("who",           "Характеристика пользователя"),
        BotCommand("anon",          "Анонимное сообщение"),
        BotCommand("gossip",        "Анонимная сплетня"),
        BotCommand("links",         "Полезные ссылки"),
        BotCommand("map",           "Карта СССР в Аликанте"),
    ])

    scheduler = AsyncIOScheduler(timezone=MADRID_TZ)

    # Поздравления — полночь
    scheduler.add_job(check_birthdays, "cron", hour=0, minute=0, args=[app.bot])

    # Авто-сводка — 09:00 (если включена)
    if MAIN_CHAT_ID:
        scheduler.add_job(
            send_auto_summary, "cron",
            hour=9, minute=0,
            args=[app.bot, MAIN_CHAT_ID]
        )

    # Чистка истёкших анон-сообщений — каждый час
    scheduler.add_job(cleanup_expired_anon_messages, "interval", hours=1)

    # Чистка старых сплетен — раз в сутки
    scheduler.add_job(cleanup_old_gossips, "cron", hour=3, minute=0)

    # Пингер — каждые 10 минут
    scheduler.add_job(self_ping, "interval", minutes=10)

    scheduler.start()

    await app.initialize()
    await app.start()
    await app.updater.start_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )
    logger.info("UniHelper started ✅")

    try:
        await asyncio.Event().wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
