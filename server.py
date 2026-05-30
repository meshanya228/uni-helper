import asyncio
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from aiohttp import web, ClientSession, ClientTimeout
from telegram import Update, BotCommand, BotCommandScopeAllGroupChats
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ChatMemberHandler, CallbackQueryHandler)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from handlers.voice    import handle_voice_transcribe
from handlers.birthday import cmd_birthday, check_birthdays
from handlers.welcome  import handle_new_member
from handlers.ai       import (cmd_ai, cmd_image, cmd_summary, cmd_summary_toggle,
                                cmd_who, send_auto_summary)
from handlers.collect  import handle_any_message
from handlers.links    import cmd_links
from handlers.anon     import cmd_anon, handle_anon_callback
from handlers.gossip   import cmd_gossip
from handlers.map_cmd  import cmd_map
from utils.db          import (init_db, cleanup_expired_anon_messages,
                                cleanup_old_gossips)
from services.gemini   import ensure_bg_worker, check_api_health

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)
MADRID_TZ = ZoneInfo("Europe/Madrid")

MAIN_CHAT_ID = int(os.environ.get("MAIN_CHAT_ID", "0"))
_known_group_chats: set[int] = set()


# ─── HTTP keepalive ────────────────────────────────────────────────────────────

async def health(request):
    return web.Response(text="UniHelper is alive 🤖", status=200)

async def start_http_server():
    app = web.Application()
    app.router.add_get("/",       health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 7860))
    await web.TCPSite(runner, "0.0.0.0", port).start()
    logger.info(f"HTTP server started on :{port}")


# ─── Self-pinger ──────────────────────────────────────────────────────────────

async def self_ping():
    """Каждые 10 минут. Окно тишины 04:00–06:00 по Мадриду."""
    now = datetime.now(MADRID_TZ)
    if 4 <= now.hour < 6:
        return
    base = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not base:
        port = int(os.environ.get("PORT", 7860))
        base = f"http://localhost:{port}"
    url = f"{base}/health"
    try:
        timeout = ClientTimeout(total=15)
        async with ClientSession(timeout=timeout) as s:
            async with s.get(url) as r:
                logger.debug(f"Self-ping {r.status}")
    except Exception as e:
        logger.debug(f"Self-ping skipped: {e}")


# ─── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context):
    text = (
        "Привет! Я UniHelper — бот UAшников 🎓\n\n"
        "Команды:\n"
        "/ai [вопрос] — спросить у ИИ (Groq)\n"
        "/image [описание] — сгенерировать картинку\n"
        "/voice — расшифровать голосовое (ответь на него)\n"
        "/birthday ДД.ММ — сохранить день рождения\n"
        "/summary — сводка за вчера\n"
        "/summarytoggle — вкл/выкл авто-сводку\n"
        "/who @user — характеристика пользователя\n"
        "/anon @user текст — анонимное сообщение\n"
        "/gossip текст — анонимная сплетня\n"
        "/links — полезные ссылки\n"
        "/map — карта СССР в Аликанте"
    )
    await update.message.reply_text(text)


# ─── Трекер групп ─────────────────────────────────────────────────────────────

async def track_group(update: Update, context):
    if update.message and update.message.chat_id:
        from telegram.constants import ChatType
        if update.message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
            _known_group_chats.add(update.message.chat_id)


# ─── Авто-сводка по всем группам ──────────────────────────────────────────────

async def auto_summary_all(bot):
    targets: set[int] = set()
    if MAIN_CHAT_ID:
        targets.add(MAIN_CHAT_ID)
    targets.update(_known_group_chats)
    for chat_id in targets:
        try:
            await send_auto_summary(bot, chat_id)
        except Exception as e:
            logger.error(f"auto_summary error for {chat_id}: {e}")


# ─── Application ──────────────────────────────────────────────────────────────

async def build_application() -> Application:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app   = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start",         cmd_start))
    app.add_handler(CommandHandler("help",          cmd_start))
    app.add_handler(CommandHandler("voice",         handle_voice_transcribe))
    app.add_handler(CommandHandler("birthday",      cmd_birthday))
    app.add_handler(CommandHandler("ai",            cmd_ai))
    app.add_handler(CommandHandler("image",         cmd_image))
    app.add_handler(CommandHandler("summary",       cmd_summary))
    app.add_handler(CommandHandler("summarytoggle", cmd_summary_toggle))
    app.add_handler(CommandHandler("who",           cmd_who))
    app.add_handler(CommandHandler("links",         cmd_links))
    app.add_handler(CommandHandler("anon",          cmd_anon))
    app.add_handler(CommandHandler("gossip",        cmd_gossip))
    app.add_handler(CommandHandler("map",           cmd_map))

    app.add_handler(CallbackQueryHandler(handle_anon_callback, pattern=r"^anon:\d+$"))
    app.add_handler(ChatMemberHandler(handle_new_member, ChatMemberHandler.CHAT_MEMBER))

    app.add_handler(MessageHandler(
        filters.ALL & ~filters.COMMAND, track_group), group=0)
    app.add_handler(MessageHandler(
        filters.ALL & ~filters.COMMAND, handle_any_message), group=1)

    return app


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    await start_http_server()
    await init_db()
    ensure_bg_worker()

    api_status = await check_api_health()
    logger.info(f"Gemini API status: {api_status}")

    # Groq API key check
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        logger.info("Groq API key: ✅ present")
    else:
        logger.warning("Groq API key: ❌ GROQ_API_KEY not set")

    app = await build_application()

    commands = [
        BotCommand("start",         "Информация о боте"),
        BotCommand("help",          "Справка по командам"),
        BotCommand("voice",         "Расшифровать голосовое (ответь на него)"),
        BotCommand("birthday",      "Сохранить ДР — пример: /birthday 15.03"),
        BotCommand("ai",            "Спросить у ИИ — пример: /ai расскажи анекдот"),
        BotCommand("image",         "Сгенерировать картинку — пример: /image закат"),
        BotCommand("summary",       "Сводка за вчера"),
        BotCommand("summarytoggle", "Вкл/выкл авто-сводку"),
        BotCommand("who",           "Характеристика — пример: /who @username"),
        BotCommand("anon",          "Анонимно — пример: /anon @user привет"),
        BotCommand("gossip",        "Сплетня — пример: /gossip слышал что..."),
        BotCommand("links",         "Полезные ссылки"),
        BotCommand("map",           "Карта СССР в Аликанте"),
    ]

    await app.bot.set_my_commands(commands)
    await app.bot.set_my_commands(
        commands, scope=BotCommandScopeAllGroupChats())

    scheduler = AsyncIOScheduler(timezone=MADRID_TZ)
    scheduler.add_job(check_birthdays,               "cron",     hour=0,  minute=0,  args=[app.bot])
    scheduler.add_job(auto_summary_all,              "cron",     hour=9,  minute=0,  args=[app.bot])
    scheduler.add_job(cleanup_expired_anon_messages, "interval", hours=1)
    scheduler.add_job(cleanup_old_gossips,           "cron",     hour=3,  minute=30)
    scheduler.add_job(self_ping,                     "interval", minutes=10)
    scheduler.start()

    await app.initialize()
    await app.start()

    await app.updater.start_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True)

    logger.info("UniHelper started ✅ (Groq + Gemini hybrid)")

    try:
        await asyncio.Event().wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
