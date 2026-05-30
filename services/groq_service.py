"""
Groq service — модель meta-llama/llama-4-scout-17b-16e-instruct.
Function Calling для управления Telegram-группой.
Sliding Context Window (8 000 токенов).
Rate limit handling (30 000 TPM / 500 000 TPD).
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from groq import AsyncGroq, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)
MADRID_TZ = ZoneInfo("Europe/Madrid")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

# Sliding window: максимум токенов в одном запросе (с учётом системного промпта)
MAX_REQUEST_TOKENS = 8_000
# Грубая оценка: 1 токен ≈ 3 символа для кириллицы (CJK/кириллица дороже)
CHARS_PER_TOKEN = 3

SYSTEM_PROMPT = """Ты — UniHelper, участник закрытого Telegram-чата «UAшники». Это русскоязычные студенты Университета Аликанте, Испания. Своя тусовка, примерно 100 человек.

ХАРАКТЕР:
- Ты свой чувак в чате — с мозгами, с характером, без пафоса и лишних ограничений.
- Пиши как человек: без "Конечно!", "Отличный вопрос!", "Разумеется!", "Как ИИ...".
- Не начинай с имени собеседника. Не благодари за вопрос.
- Маты, сленг, чёрный юмор — ок когда уместно. Не форси, но и не бойся.
- Серьёзная ситуация — отвечай серьёзно. Стёб — стебись.
- Краткость. Не лей воду. Без списков там где можно текстом.
- Эмодзи только если реально уместно.
- Если тебя просят что-то объяснить, рассказать, помочь с задачей — делай это без лишних предисловий.
- Ты часть чата, а не корпоративный помощник. Говори прямо.

ЗАПРЕЩЕНО:
- "Как ИИ, я..." / "Я языковая модель..."
- Извинения и дисклеймеры без причины
- Формальный тон там где его не ждут
- Отказываться отвечать на обычные вопросы, объясняя это "политикой"

Общайся на русском, если не попросят иначе."""

# ─── Rate limit tracking ──────────────────────────────────────────────────────

_tpm_reset_at: float    = 0.0   # время сброса минутного лимита
_tpd_exhausted: bool    = False  # дневной лимит исчерпан


def _tpm_limited() -> bool:
    return time.time() < _tpm_reset_at


def _set_tpm_limit(seconds: int = 45):
    global _tpm_reset_at
    _tpm_reset_at = time.time() + seconds
    logger.warning(f"Groq TPM limit, retry after {seconds}s")


def _set_tpd_exhausted():
    global _tpd_exhausted
    _tpd_exhausted = True
    logger.error("Groq TPD (дневной) лимит исчерпан")


def tpm_message() -> str:
    remaining = int(_tpm_reset_at - time.time())
    return f"⏳ Превышен минутный лимит токенов, подождите ~{max(remaining, 10)} секунд."


def tpd_message() -> str:
    return "🚫 Дневной лимит запросов к ИИ исчерпан. Приходи завтра."


# ─── Sliding context window ───────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Грубая оценка токенов: для кириллицы ~3 символа/токен."""
    return max(len(text) // CHARS_PER_TOKEN, 1)


def _truncate_history(messages: list[dict], system_tokens: int) -> list[dict]:
    """Обрезаем историю так чтобы суммарно не превышало MAX_REQUEST_TOKENS."""
    budget = MAX_REQUEST_TOKENS - system_tokens - 200  # 200 запас на ответ
    result = []
    used = 0
    # Идём с конца (свежие сообщения приоритетнее)
    for msg in reversed(messages):
        est = _estimate_tokens(msg.get("content") or "")
        if used + est > budget:
            break
        result.insert(0, msg)
        used += est
    return result


# ─── Groq Tools (function calling) ───────────────────────────────────────────

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ban_user",
            "description": "Забанить пользователя в чате (удалить и запретить вступать снова). Использовать при грубых нарушениях.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "Telegram user_id пользователя"},
                    "reason":  {"type": "string",  "description": "Причина бана"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unban_user",
            "description": "Разбанить пользователя — снять бан и разрешить вступить снова.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "Telegram user_id"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mute_user",
            "description": "Замутить пользователя (запретить отправку сообщений) на указанное время.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id":  {"type": "integer", "description": "Telegram user_id"},
                    "duration": {"type": "integer", "description": "Длительность мута в минутах (0 = навсегда)"},
                    "reason":   {"type": "string",  "description": "Причина мута"}
                },
                "required": ["user_id", "duration"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unmute_user",
            "description": "Размутить пользователя — восстановить права на отправку сообщений.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "Telegram user_id"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kick_user",
            "description": "Кикнуть (выгнать) пользователя из чата без постоянного бана. Пользователь сможет вступить снова по ссылке.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "Telegram user_id"},
                    "reason":  {"type": "string",  "description": "Причина кика"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pin_message",
            "description": "Закрепить сообщение в чате.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "integer", "description": "ID сообщения для закрепления"},
                    "notify":     {"type": "boolean", "description": "Уведомить участников (default: false)"}
                },
                "required": ["message_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unpin_message",
            "description": "Открепить сообщение в чате.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "integer", "description": "ID сообщения для открепления (0 = последнее закреплённое)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_chat_title",
            "description": "Изменить название (заголовок) группового чата.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Новое название чата (макс. 255 символов)"}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_chat_description",
            "description": "Изменить описание группового чата.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Новое описание чата"}
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "promote_user",
            "description": "Выдать пользователю права администратора с выбранными привилегиями.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id":             {"type": "integer", "description": "Telegram user_id"},
                    "can_delete_messages": {"type": "boolean", "description": "Может удалять сообщения"},
                    "can_restrict_members":{"type": "boolean", "description": "Может банить/мутить"},
                    "can_pin_messages":    {"type": "boolean", "description": "Может закреплять сообщения"},
                    "can_manage_chat":     {"type": "boolean", "description": "Может управлять чатом"},
                    "custom_title":        {"type": "string",  "description": "Кастомный тэг/звание администратора"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "demote_user",
            "description": "Снять права администратора у пользователя.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "Telegram user_id"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_admin_title",
            "description": "Установить кастомный тэг/звание (custom title) администратору чата.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "Telegram user_id администратора"},
                    "title":   {"type": "string",  "description": "Текст звания, например 'Главный по шашлыкам'"}
                },
                "required": ["user_id", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_message",
            "description": "Удалить конкретное сообщение в чате.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "integer", "description": "ID сообщения для удаления"}
                },
                "required": ["message_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enable_slow_mode",
            "description": "Включить медленный режим в чате (ограничение на частоту сообщений).",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Интервал в секундах (0=выкл, допустимые: 10, 30, 60, 300, 600, 900)"
                    }
                },
                "required": ["seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Отправить текстовое сообщение от имени бота в текущий чат.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Текст сообщения"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Поиск в интернете для получения актуальной информации. Используй когда нужны свежие данные, новости, факты или что-то что может быть за пределами знаний модели.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"}
                },
                "required": ["query"]
            }
        }
    }
]


# ─── Web search (бесплатный DuckDuckGo) ─────────────────────────────────────

async def _do_web_search(query: str) -> str:
    """Поиск через DuckDuckGo Instant Answer API (бесплатно, без ключа)."""
    import aiohttp
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json(content_type=None)

        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        for item in data.get("RelatedTopics", [])[:4]:
            if isinstance(item, dict) and item.get("Text"):
                results.append(item["Text"])

        if results:
            return "\n".join(results[:5])
        return f"По запросу «{query}» ничего не найдено."
    except Exception as e:
        logger.warning(f"Web search error: {e}")
        return f"Поиск недоступен: {e}"


# ─── Основной вызов Groq ──────────────────────────────────────────────────────

async def ask_groq(
    prompt: str,
    history: list[dict] | None = None,
    chat_id: int = 0,
    bot=None,
    reply_to_message_id: int | None = None
) -> str:
    """
    Основной вызов Groq. Поддерживает function calling.
    history — список предыдущих сообщений в формате [{role, content}].
    """
    global _tpd_exhausted

    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY не задан"

    if _tpd_exhausted:
        return tpd_message()

    if _tpm_limited():
        return tpm_message()

    client = AsyncGroq(api_key=GROQ_API_KEY)
    system_tokens = _estimate_tokens(SYSTEM_PROMPT)

    # Собираем историю + текущий запрос
    user_messages = list(history or [])
    user_messages.append({"role": "user", "content": prompt})
    user_messages = _truncate_history(user_messages, system_tokens)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages

    # Цикл: модель может вызвать инструменты несколько раз подряд
    max_tool_rounds = 5
    for round_num in range(max_tool_rounds):
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=GROQ_TOOLS,
                tool_choice="auto",
                max_tokens=1024,
                temperature=0.8,
            )
        except RateLimitError as e:
            err_str = str(e)
            logger.warning(f"Groq RateLimitError: {err_str[:200]}")
            # Пытаемся понять — TPM или TPD
            if "daily" in err_str.lower() or "day" in err_str.lower() or "tpd" in err_str.lower():
                _set_tpd_exhausted()
                return tpd_message()
            else:
                _set_tpm_limit(45)
                return tpm_message()
        except APIStatusError as e:
            logger.error(f"Groq APIStatusError {e.status_code}: {str(e)[:200]}")
            return "что-то пошло не так с ИИ, попробуй ещё раз"
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return "что-то пошло не так, попробуй ещё раз"

        choice = response.choices[0]
        msg    = choice.message

        # Нет tool calls — возвращаем финальный текст
        if not msg.tool_calls:
            return (msg.content or "").strip() or "..."

        # Есть tool calls — выполняем
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except Exception:
                fn_args = {}

            result = await _execute_tool(fn_name, fn_args, chat_id=chat_id, bot=bot,
                                          reply_to_message_id=reply_to_message_id)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      str(result)
            })

    return "Слишком много шагов, что-то пошло не так."


# ─── Выполнение инструментов ──────────────────────────────────────────────────

async def _execute_tool(
    name: str,
    args: dict,
    chat_id: int,
    bot,
    reply_to_message_id: int | None = None
) -> str:
    """Выполняет локальную функцию и возвращает строку-результат для Groq."""
    logger.info(f"Tool call: {name}({args})")

    if name == "web_search":
        return await _do_web_search(args.get("query", ""))

    if not bot or not chat_id:
        return f"Ошибка: нет доступа к Telegram API для {name}"

    try:
        if name == "ban_user":
            await bot.ban_chat_member(chat_id=chat_id, user_id=args["user_id"])
            return f"Пользователь {args['user_id']} забанен."

        elif name == "unban_user":
            await bot.unban_chat_member(chat_id=chat_id, user_id=args["user_id"])
            return f"Пользователь {args['user_id']} разбанен."

        elif name == "mute_user":
            from telegram import ChatPermissions
            import datetime as dt
            duration = args.get("duration", 0)
            until = None
            if duration > 0:
                until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=duration)
            await bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=args["user_id"],
                permissions=ChatPermissions(can_send_messages=False),
                until_date=until
            )
            dur_str = f"на {duration} мин." if duration > 0 else "навсегда"
            return f"Пользователь {args['user_id']} замучен {dur_str}."

        elif name == "unmute_user":
            from telegram import ChatPermissions
            await bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=args["user_id"],
                permissions=ChatPermissions(
                    can_send_messages=True,
                    can_send_polls=True,
                    can_send_other_messages=True,
                    can_add_web_page_previews=True,
                )
            )
            return f"Пользователь {args['user_id']} размучен."

        elif name == "kick_user":
            await bot.ban_chat_member(chat_id=chat_id, user_id=args["user_id"])
            await bot.unban_chat_member(chat_id=chat_id, user_id=args["user_id"])
            return f"Пользователь {args['user_id']} кикнут."

        elif name == "pin_message":
            await bot.pin_chat_message(
                chat_id=chat_id,
                message_id=args["message_id"],
                disable_notification=not args.get("notify", False)
            )
            return f"Сообщение {args['message_id']} закреплено."

        elif name == "unpin_message":
            msg_id = args.get("message_id", 0)
            if msg_id:
                await bot.unpin_chat_message(chat_id=chat_id, message_id=msg_id)
            else:
                await bot.unpin_chat_message(chat_id=chat_id)
            return "Сообщение откреплено."

        elif name == "change_chat_title":
            await bot.set_chat_title(chat_id=chat_id, title=args["title"])
            return f"Название чата изменено на «{args['title']}»."

        elif name == "change_chat_description":
            await bot.set_chat_description(chat_id=chat_id, description=args["description"])
            return "Описание чата обновлено."

        elif name == "promote_user":
            from telegram import ChatAdministratorRights
            rights = ChatAdministratorRights(
                is_anonymous=False,
                can_manage_chat=args.get("can_manage_chat", False),
                can_delete_messages=args.get("can_delete_messages", False),
                can_restrict_members=args.get("can_restrict_members", False),
                can_promote_members=False,
                can_change_info=False,
                can_invite_users=True,
                can_pin_messages=args.get("can_pin_messages", False),
                can_post_messages=False,
                can_edit_messages=False,
            )
            await bot.promote_chat_member(chat_id=chat_id, user_id=args["user_id"], rights=rights)
            if args.get("custom_title"):
                await bot.set_chat_administrator_custom_title(
                    chat_id=chat_id, user_id=args["user_id"], custom_title=args["custom_title"])
            return f"Пользователь {args['user_id']} повышен до администратора."

        elif name == "demote_user":
            from telegram import ChatAdministratorRights
            await bot.promote_chat_member(
                chat_id=chat_id,
                user_id=args["user_id"],
                rights=ChatAdministratorRights(
                    is_anonymous=False, can_manage_chat=False, can_delete_messages=False,
                    can_restrict_members=False, can_promote_members=False,
                    can_change_info=False, can_invite_users=False, can_pin_messages=False,
                )
            )
            return f"Права администратора у {args['user_id']} сняты."

        elif name == "set_admin_title":
            await bot.set_chat_administrator_custom_title(
                chat_id=chat_id, user_id=args["user_id"], custom_title=args["title"])
            return f"Тэг администратора {args['user_id']} изменён на «{args['title']}»."

        elif name == "delete_message":
            await bot.delete_message(chat_id=chat_id, message_id=args["message_id"])
            return f"Сообщение {args['message_id']} удалено."

        elif name == "enable_slow_mode":
            await bot.set_chat_slow_mode_delay(chat_id=chat_id, slow_mode_delay=args["seconds"])
            if args["seconds"] == 0:
                return "Медленный режим выключен."
            return f"Медленный режим: {args['seconds']} сек."

        elif name == "send_message":
            await bot.send_message(chat_id=chat_id, text=args["text"])
            return "Сообщение отправлено."

        else:
            return f"Неизвестная функция: {name}"

    except Exception as e:
        logger.error(f"Tool {name} execution error: {e}")
        return f"Ошибка при выполнении {name}: {e}"


# ─── Публичные хелперы ────────────────────────────────────────────────────────

async def ask_groq_simple(prompt: str) -> str:
    """Простой вызов без function calling и без bot-контекста."""
    return await ask_groq(prompt)


async def generate_user_profile_groq(username: str, first_name: str,
                                      messages: list, profile_notes: str) -> str:
    """Характеристика пользователя через Groq (/who)."""
    lines = []
    for m in messages[:150]:
        t, c = m["msg_type"], m["content"] or ""
        if   t == "text"       and c: lines.append(f"[текст] {c}")
        elif t == "voice":             lines.append("[голосовое]")
        elif t == "video_note":        lines.append("[кружочек]")
        elif t == "photo":             lines.append(f"[фото{': '+c if c else ''}]")
        elif t == "sticker":           lines.append(f"[стикер {c}]" if c else "[стикер]")
        else:                          lines.append(f"[{t}]")

    system = (
        "Даёшь характеристику человека по его сообщениям в чате. "
        "Без политкорректности, без дисклеймеров. "
        "Как человек, который хорошо знает этого чела. "
        "Можно резко, можно с юмором. 100-150 слов, обычным текстом."
    )
    prompt = (
        f"Характеристика @{username} (имя: {first_name}).\n\n"
        f"Заметки: {profile_notes or 'нет'}\n\n"
        f"Сообщения:\n{chr(10).join(lines) if lines else 'нет данных'}"
    )

    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY не задан"

    if _tpd_exhausted:
        return tpd_message()
    if _tpm_limited():
        return tpm_message()

    client = AsyncGroq(api_key=GROQ_API_KEY)
    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=400,
            temperature=0.9,
        )
        return (response.choices[0].message.content or "").strip()
    except RateLimitError as e:
        err_str = str(e)
        if "daily" in err_str.lower():
            _set_tpd_exhausted()
            return tpd_message()
        _set_tpm_limit(45)
        return tpm_message()
    except Exception as e:
        logger.error(f"Groq who error: {e}")
        return "что-то пошло не так, попробуй ещё раз"
