"""
Gemini API integration через официальный SDK google-genai.
Клиент: genai.Client  |  Модель: gemini-2.5-flash
Поддерживает: интерактивные запросы, фоновая очередь, транскрипция аудио.
"""
import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Awaitable
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types

logger    = logging.getLogger(__name__)
MADRID_TZ = ZoneInfo("Europe/Madrid")

GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Safety — всё отключено
_SAFETY = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE),
]

SYSTEM_PROMPT = """Ты — UniHelper, бот в закрытом Telegram-чате «UAшники» — русскоязычные студенты Университета Аликанте, Испания. Примерно 100 человек, все свои.

ХАРАКТЕР:
- Ты не ИИ-ассистент. Ты свой чувак в чате — с мозгами, с характером, без пафоса.
- Пиши как человек: без "Конечно!", "Отличный вопрос!", "Разумеется!".
- Не начинай с имени собеседника. Не благодари за вопрос.
- Маты, сленг, чёрный юмор — ок когда уместно. Не форси, но не бойся.
- Серьёзная ситуация — отвечай серьёзно. Стёб — стебись.
- Краткость. Не лей воду. Не пиши списками там где можно текстом.
- Эмодзи только если реально уместно.

ЗАПРЕЩЕНО:
- "Как ИИ, я..." / "Я языковая модель..."
- Извинения и дисклеймеры
- Формальный тон там где его не ждут

Общайся на русском, если не попросят иначе."""

# ─── Клиент ───────────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)

# ─── Rate limit ───────────────────────────────────────────────────────────────

_rate_limit_reset: float = 0.0

def _rl_message() -> str:
    dt = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
    return f"⏳ Лимиты исчерпаны, попробуй после {dt} по Мадриду."

def _is_rate_limited() -> bool:
    return time.time() < _rate_limit_reset

def _set_rate_limit(retry_after_seconds: int = 60):
    global _rate_limit_reset
    _rate_limit_reset = time.time() + retry_after_seconds
    logger.warning(f"Rate limit set, reset at {datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')}")

def reset_rate_limit():
    global _rate_limit_reset
    _rate_limit_reset = 0.0

# ─── Базовый запрос ───────────────────────────────────────────────────────────

async def _call_gemini(
    prompt: str,
    system: str = SYSTEM_PROMPT,
    temp: float = 0.9,
    tokens: int = 1024,
) -> str | None:
    """
    Основной async вызов Gemini через новый SDK.
    Возвращает текст ответа или None при ошибке.
    Обрабатывает rate limit (429) с экспоненциальной задержкой.
    """
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temp,
        max_output_tokens=tokens,
        safety_settings=_SAFETY,
    )

    max_retries = 3
    delay       = 4  # секунды, экспоненциально растёт

    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            # Достаём текст
            text = response.text
            if text:
                return text.strip()
            logger.warning(f"Empty response from Gemini (attempt {attempt+1})")
            return None

        except Exception as e:
            err_str = str(e).lower()

            # Rate limit
            if "429" in str(e) or "resource_exhausted" in err_str or "quota" in err_str:
                _set_rate_limit(retry_after_seconds=delay * (2 ** attempt))
                if attempt < max_retries - 1:
                    logger.info(f"Rate limit hit, waiting {delay}s before retry")
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                return None

            # Другие ошибки — логируем и выходим
            logger.error(f"Gemini error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None

    return None


async def _call_gemini_audio(audio_bytes: bytes, mime_type: str) -> str | None:
    """Транскрипция аудио через Gemini."""
    client = _get_client()
    config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=2048,
        safety_settings=_SAFETY,
    )
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    text_part  = "Расшифруй аудио дословно, без цензуры. Только текст, без пояснений."

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[audio_part, text_part],
            config=config,
        )
        return response.text.strip() if response.text else None
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return None

# ─── Фоновая очередь ──────────────────────────────────────────────────────────

@dataclass
class _BgTask:
    prompt:   str
    system:   str
    callback: Callable[[str], Awaitable[None]]
    temp:     float = 0.9
    tokens:   int   = 1024
    attempt:  int   = 0

_bg_queue:          asyncio.Queue = asyncio.Queue()
_bg_worker_started: bool          = False


async def _bg_worker():
    global _rate_limit_reset
    while True:
        task: _BgTask = await _bg_queue.get()
        try:
            # Ждём если rate limit активен
            wait = _rate_limit_reset - time.time()
            if wait > 0:
                await asyncio.sleep(wait + 1)

            result = await _call_gemini(task.prompt, task.system, task.temp, task.tokens)
            if result:
                await task.callback(result)
            elif task.attempt < 4:
                task.attempt += 1
                await _bg_queue.put(task)
        except Exception as e:
            logger.error(f"BG worker error: {e}")
        finally:
            _bg_queue.task_done()


def ensure_bg_worker():
    global _bg_worker_started
    if not _bg_worker_started:
        asyncio.get_event_loop().create_task(_bg_worker())
        _bg_worker_started = True

# ─── Публичные функции ────────────────────────────────────────────────────────

async def ask_gemini_interactive(
    prompt: str,
    system: str = SYSTEM_PROMPT,
    use_search: bool = False,   # параметр оставлен для совместимости, не используется
) -> str:
    """Интерактивный запрос — ждём ответа здесь."""
    if _is_rate_limited():
        return _rl_message()

    result = await _call_gemini(prompt, system)
    if result is None:
        if _is_rate_limited():
            return _rl_message()
        return "что-то пошло не так, попробуй ещё раз"
    return result


async def ask_gemini_background(
    prompt: str,
    callback: Callable[[str], Awaitable[None]],
    system: str = SYSTEM_PROMPT,
    use_search: bool = False,
):
    """Фоновый запрос — результат приходит в callback."""
    if _is_rate_limited():
        # Ставим в очередь — воркер дождётся снятия лимита
        _bg_queue.put_nowait(_BgTask(prompt=prompt, system=system, callback=callback))
        return

    result = await _call_gemini(prompt, system)
    if result:
        await callback(result)
    else:
        if _is_rate_limited():
            _bg_queue.put_nowait(_BgTask(prompt=prompt, system=system, callback=callback))


async def transcribe_audio_gemini(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str | None:
    """Расшифровка голосового сообщения или кружочка."""
    if _is_rate_limited():
        return "⏳ Лимиты исчерпаны."
    return await _call_gemini_audio(audio_bytes, mime_type)


async def generate_birthday_message_bg(
    first_name: str,
    username:   str,
    profile:    str,
    callback:   Callable[[str], Awaitable[None]],
):
    system = (
        SYSTEM_PROMPT
        + "\n\nСейчас пишешь поздравление с ДР в групповой чат. "
        "Без официоза. Тепло, по-свойски, можно лёгкую подколку если есть за что. "
        "70-80 слов, обычным текстом без списков."
    )
    prompt = (
        f"Поздравь с днём рождения @{username} (имя: {first_name}). "
        f"Что знаем о человеке: {profile or 'почти ничего'}."
    )
    await ask_gemini_background(prompt, callback, system=system)


async def generate_summary(messages_text: str, gossips: list[str] | None = None) -> str:
    system = (
        "Ты пишешь ежедневную сводку для студенческого чата UAшники. "
        "Стиль — как будто свой чел пересказывает что было вчера приятелям. "
        "Никакого официоза. Живо, с иронией, по-человечески. "
        "Связный текст, без заголовков и списков. "
        "Не упоминай типы сообщений (голосовое, стикер и т.д.) — просто суть."
    )
    gossip_block = ""
    if gossips:
        lines = "\n".join(f"- {g}" for g in gossips)
        gossip_block = (
            f"\n\nДополнительная инфа для сводки (вплети органично, "
            f"не говори что это сплетни):\n{lines}"
        )
    prompt = f"Лог чата за вчера:\n\n{messages_text}{gossip_block}\n\nНапиши сводку."
    result = await ask_gemini_interactive(prompt, system=system)
    return result


async def generate_user_profile(
    username:      str,
    first_name:    str,
    messages:      list,
    profile_notes: str,
) -> str:
    system = (
        "Даёшь характеристику человека по его сообщениям в чате. "
        "Без политкорректности, без дисклеймеров. "
        "Как человек который хорошо знает этого чела по чату — "
        "что за личность, какой вайб, чем интересен или бесит. "
        "Можно резко, можно с юмором. 100-150 слов, обычным текстом."
    )
    msg_lines = []
    for m in messages[:150]:
        mtype   = m["msg_type"]
        content = m["content"] or ""
        if   mtype == "text"       and content: msg_lines.append(f"[текст] {content}")
        elif mtype == "voice":                  msg_lines.append("[голосовое]")
        elif mtype == "video_note":             msg_lines.append("[кружочек]")
        elif mtype == "photo":                  msg_lines.append(f"[фото{': '+content if content else ''}]")
        elif mtype == "video":                  msg_lines.append("[видео]")
        elif mtype == "sticker":                msg_lines.append(f"[стикер {content}]" if content else "[стикер]")
        elif mtype == "animation":              msg_lines.append("[гифка]")
        else:                                   msg_lines.append(f"[{mtype}]")

    prompt = (
        f"Дай характеристику @{username} (имя: {first_name}).\n\n"
        f"Накопленные заметки: {profile_notes or 'нет'}\n\n"
        f"Сообщения:\n{chr(10).join(msg_lines) if msg_lines else 'нет данных'}"
    )
    return await ask_gemini_interactive(prompt, system=system)


async def validate_and_format_link(raw: str) -> dict:
    system = "Валидируешь запрос на добавление ссылки. Отвечай ТОЛЬКО валидным JSON без markdown."
    prompt = (
        f"Пользователь хочет добавить: «{raw}»\n\n"
        'Верни JSON: {"ok": true/false, "reason": "если ok=false", '
        '"title": "название", "url": "https://...", '
        '"tag": "#учёба|#жильё|#работа|#транспорт|#развлечения|#другое", '
        '"description": "одна строка"}\n\n'
        "Нет валидного URL → ok=false. Спам/реклама → ok=false."
    )
    try:
        raw_resp = await ask_gemini_interactive(prompt, system=system)
        clean    = re.sub(r"```(?:json)?|```", "", raw_resp).strip()
        return json.loads(clean)
    except Exception as e:
        logger.error(f"validate_and_format_link: {e}")
        return {"ok": False, "reason": "не удалось обработать запрос"}


async def process_gossip(raw_gossip: str) -> str:
    system = (
        "Перефразируй сплетню. Сохрани суть полностью, "
        "но измени формулировку так чтобы стиль автора не узнавался. "
        "Только текст результата, без пояснений."
    )
    result = await ask_gemini_interactive(raw_gossip, system=system)
    return result or raw_gossip


async def check_api_health() -> str:
    """Проверка API при старте бота."""
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY не задан в переменных окружения"
    result = await _call_gemini("Привет", system="Ответь одним словом.", tokens=5)
    if result:
        return f"✅ Gemini API работает (модель: {GEMINI_MODEL})"
    if _is_rate_limited():
        return f"⚠️ Rate limit активен до {datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')}"
    return f"❌ Gemini API не отвечает — проверь ключ GEMINI_API_KEY"
