"""
Приветствие новых участников — только живых людей, не ботов.
Отправляется в Флудилку.
"""
import logging
from telegram import Update, ChatMemberUpdated
from telegram.ext import ContextTypes
from telegram.constants import ChatMemberStatus

logger = logging.getLogger(__name__)
MAIN_TOPIC_ID = 1  # Флудилка


async def handle_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result: ChatMemberUpdated = update.chat_member
    if not result:
        return

    # Проверяем что именно вступил (не вышел, не получил права)
    old_status = result.old_chat_member.status
    new_status = result.new_chat_member.status

    # Человек вступил = был not-member/left/banned → стал member/restricted/admin
    joined = (
        old_status in (ChatMemberStatus.LEFT, ChatMemberStatus.BANNED,
                       ChatMemberStatus.RESTRICTED)
        and new_status in (ChatMemberStatus.MEMBER, ChatMemberStatus.RESTRICTED,
                           ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER)
    )
    if not joined:
        return

    user = result.new_chat_member.user
    if user.is_bot:
        return

    chat_id = result.chat.id

    text = (
        f"👋 Привет, {user.mention_html()}!\n\n"
        f"Добро пожаловать в UAшники — закрытый чат русскоязычных студентов Университета Аликанте 🎓\n\n"
        f"Расскажи немного о себе: на каком факультете и направлении учишься, "
        f"откуда приехал?"
    )

    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            message_thread_id=MAIN_TOPIC_ID)
    except Exception:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML")
        except Exception as e:
            logger.error(f"Welcome send failed: {e}")
