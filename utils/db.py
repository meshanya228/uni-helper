import os
import asyncpg
import logging
import datetime as dt

logger = logging.getLogger(__name__)
_pool = None

async def get_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=5)
    return _pool

async def init_db():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS birthdays (
                user_id BIGINT PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                birth_day INTEGER NOT NULL,
                birth_month INTEGER NOT NULL,
                chat_id BIGINT NOT NULL,
                user_profile TEXT DEFAULT ''
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary_log (
                chat_id BIGINT NOT NULL,
                date DATE NOT NULL,
                PRIMARY KEY (chat_id, date)
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS message_log (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                user_id BIGINT,
                username TEXT,
                first_name TEXT,
                msg_type TEXT NOT NULL,
                content TEXT,
                logged_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id BIGINT PRIMARY KEY,
                chat_id BIGINT,
                username TEXT,
                first_name TEXT,
                profile_notes TEXT DEFAULT '',
                message_count INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS useful_links (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                title TEXT NOT NULL,
                url TEXT DEFAULT '',
                tag TEXT DEFAULT '#другое',
                description TEXT DEFAULT '',
                added_by_id BIGINT,
                added_by_name TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS anon_messages (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                sender_id BIGINT NOT NULL,
                recipient_usernames TEXT[] NOT NULL,
                message_text TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS gossips (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                gossip_text TEXT NOT NULL,
                date DATE NOT NULL DEFAULT (NOW() AT TIME ZONE 'Europe/Madrid')::date,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_settings (
                chat_id BIGINT PRIMARY KEY,
                auto_summary BOOLEAN DEFAULT TRUE,
                summary_toggle_count INTEGER DEFAULT 0,
                last_toggle_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
    logger.info("Tables ensured")

# ─── Birthdays ────────────────────────────────────────────────────────────────

async def save_birthday(user_id, username, first_name, day, month, chat_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO birthdays (user_id, username, first_name, birth_day, birth_month, chat_id)
            VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT (user_id) DO UPDATE
            SET username=$2, first_name=$3, birth_day=$4, birth_month=$5, chat_id=$6
        """, user_id, username, first_name, day, month, chat_id)

async def get_todays_birthdays(day, month):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch("""
            SELECT user_id, username, first_name, chat_id, user_profile
            FROM birthdays WHERE birth_day=$1 AND birth_month=$2
        """, day, month)

async def update_birthday_profile(user_id, profile):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE birthdays SET user_profile=$2 WHERE user_id=$1", user_id, profile)

async def get_birthday_by_user(user_id):
    """Получить запись ДР пользователя если есть."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT birth_day, birth_month FROM birthdays WHERE user_id=$1",
            user_id)



# ─── Message log ──────────────────────────────────────────────────────────────

async def log_message(chat_id, user_id, username, first_name, msg_type, content):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO message_log (chat_id, user_id, username, first_name, msg_type, content)
            VALUES ($1,$2,$3,$4,$5,$6)
        """, chat_id, user_id, username, first_name, msg_type, content)

async def get_yesterday_messages(chat_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch("""
            SELECT username, first_name, msg_type, content, logged_at
            FROM message_log
            WHERE chat_id=$1
              AND logged_at::date =
                  (NOW() AT TIME ZONE 'Europe/Madrid')::date - INTERVAL '1 day'
            ORDER BY logged_at
        """, chat_id)

async def get_user_messages(chat_id, user_id, limit=200):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch("""
            SELECT msg_type, content, logged_at
            FROM message_log
            WHERE chat_id=$1 AND user_id=$2
            ORDER BY logged_at DESC LIMIT $3
        """, chat_id, user_id, limit)

async def cleanup_old_messages(chat_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM message_log WHERE chat_id=$1 AND logged_at < NOW() - INTERVAL '2 days'",
            chat_id)

# ─── Summary log ──────────────────────────────────────────────────────────────

async def check_summary_used(chat_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT 1 FROM daily_summary_log
            WHERE chat_id=$1 AND date=(NOW() AT TIME ZONE 'Europe/Madrid')::date
        """, chat_id)
        return row is not None

async def mark_summary_used(chat_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO daily_summary_log (chat_id, date)
            VALUES ($1, (NOW() AT TIME ZONE 'Europe/Madrid')::date)
            ON CONFLICT DO NOTHING
        """, chat_id)

# ─── User profiles ────────────────────────────────────────────────────────────

async def upsert_user_profile(user_id, chat_id, username, first_name, notes_addition):
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT profile_notes, message_count FROM user_profiles WHERE user_id=$1", user_id)
        if existing:
            new_count = (existing["message_count"] or 0) + 1
            current_notes = existing["profile_notes"] or ""
            new_notes = current_notes
            if notes_addition:
                new_notes = (current_notes + "\n" + notes_addition).strip()[-2000:]
            await conn.execute("""
                UPDATE user_profiles
                SET username=$2, first_name=$3, message_count=$4, profile_notes=$5,
                    updated_at=NOW(), chat_id=$6
                WHERE user_id=$1
            """, user_id, username, first_name, new_count, new_notes, chat_id)
        else:
            await conn.execute("""
                INSERT INTO user_profiles
                    (user_id, chat_id, username, first_name, profile_notes, message_count)
                VALUES ($1,$2,$3,$4,$5,1)
            """, user_id, chat_id, username, first_name, notes_addition or "")

async def get_user_profile(user_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT profile_notes FROM user_profiles WHERE user_id=$1", user_id)
        return (row["profile_notes"] or "") if row else ""

async def get_user_by_username(chat_id, username):
    """username — уже без @ и в нижнем регистре."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow("""
            SELECT user_id, first_name, username, message_count
            FROM user_profiles
            WHERE chat_id=$1 AND LOWER(COALESCE(username,''))=$2
        """, chat_id, username)

# ─── Useful links ─────────────────────────────────────────────────────────────

async def add_link(chat_id, title, url, tag, description, added_by_id, added_by_name):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("""
            INSERT INTO useful_links
                (chat_id, title, url, tag, description, added_by_id, added_by_name)
            VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id
        """, chat_id, title, url, tag, description, added_by_id, added_by_name)

async def get_all_links(chat_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch("""
            SELECT id, title, url, tag, description, added_by_name
            FROM useful_links WHERE chat_id=$1 ORDER BY tag, created_at DESC
        """, chat_id)

async def get_link_by_id(chat_id, link_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT id, title, added_by_id FROM useful_links WHERE chat_id=$1 AND id=$2",
            chat_id, link_id)

async def delete_link(chat_id, link_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM useful_links WHERE chat_id=$1 AND id=$2", chat_id, link_id)

# ─── Anon messages ────────────────────────────────────────────────────────────

async def save_anon_message(chat_id, sender_id, recipient_usernames, message_text):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("""
            INSERT INTO anon_messages
                (chat_id, sender_id, recipient_usernames, message_text, expires_at)
            VALUES ($1,$2,$3,$4, NOW() + INTERVAL '24 hours')
            RETURNING id
        """, chat_id, sender_id, recipient_usernames, message_text)

async def get_anon_message(msg_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow("""
            SELECT id, recipient_usernames, message_text, expires_at
            FROM anon_messages
            WHERE id=$1 AND expires_at > NOW()
        """, msg_id)

async def cleanup_expired_anon_messages():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM anon_messages WHERE expires_at <= NOW()")

# ─── Gossips ──────────────────────────────────────────────────────────────────

async def save_gossip(chat_id, gossip_text):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO gossips (chat_id, gossip_text) VALUES ($1,$2)",
            chat_id, gossip_text)

async def get_yesterdays_gossips(chat_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch("""
            SELECT gossip_text FROM gossips
            WHERE chat_id=$1
              AND date = (NOW() AT TIME ZONE 'Europe/Madrid')::date - INTERVAL '1 day'
            ORDER BY created_at
        """, chat_id)

async def cleanup_old_gossips():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM gossips WHERE created_at < NOW() - INTERVAL '3 days'")

# ─── Chat settings ────────────────────────────────────────────────────────────

async def get_auto_summary(chat_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT auto_summary FROM chat_settings WHERE chat_id=$1", chat_id)
        return row["auto_summary"] if row else True

async def toggle_auto_summary(chat_id):
    """Переключить авто-сводку. None = лимит переключений превышен."""
    pool = await get_pool()
    now_utc = dt.datetime.now(dt.timezone.utc)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT auto_summary, summary_toggle_count, last_toggle_at "
            "FROM chat_settings WHERE chat_id=$1", chat_id)
        if row is None:
            await conn.execute("""
                INSERT INTO chat_settings
                    (chat_id, auto_summary, summary_toggle_count, last_toggle_at)
                VALUES ($1, FALSE, 1, NOW())
            """, chat_id)
            return False

        last = row["last_toggle_at"]
        if last.tzinfo is None:
            last = last.replace(tzinfo=dt.timezone.utc)
        count = 0 if (now_utc - last).total_seconds() > 86400 \
                  else (row["summary_toggle_count"] or 0)
        if count >= 10:
            return None

        new_val = not row["auto_summary"]
        await conn.execute("""
            UPDATE chat_settings
            SET auto_summary=$2, summary_toggle_count=$3, last_toggle_at=NOW()
            WHERE chat_id=$1
        """, chat_id, new_val, count + 1)
        return new_val
