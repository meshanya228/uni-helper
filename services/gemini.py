"""
Gemini API — официальный SDK google-genai.
Клиент: genai.Client(api_key=...)
Модель: gemini-3.5-flash
"""
import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Awaitable
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types

logger    = logging.getLogger(__name__)
MADRID_TZ = ZoneInfo("Europe/Madrid")

GEMINI_MODEL   = "gemini-3.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Safety — всё отключено
_SAFETY = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
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

# ─── Rate limit ───────────────────────────────────────────────────────────────

_rate_limit_reset: float = 0.0

def _is_rate_limited() -> bool:
    return time.time() < _rate_limit_reset

def _set_rate_limit(seconds: int = 60):
    global _rate_limit_reset
    _rate_limit_reset = time.time() + seconds
    t = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
    logger.warning(f"Rate limit: reset at {t} Madrid")

def _rl_message() -> str:
    t = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
    return f"⏳ Лимиты исчерпаны, попробуй после {t} по Мадриду."

def reset_rate_limit():
    global _rate_limit_reset
    _rate_limit_reset = 0.0

# ─── Клиент ───────────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)

# ─── Базовый запрос ───────────────────────────────────────────────────────────

async def _call_gemini(prompt: str,
                       system: str = SYSTEM_PROMPT,
                       temp: float = 0.9,
                       tokens: int = 1024) -> str | None:
    """
    Вызов Gemini API. Возвращает текст или None.
    Rate limit устанавливается ТОЛЬКО при финальном 429 после всех попыток.
    """
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temp,
        max_output_tokens=tokens,
        safety_settings=_SAFETY,
    )

    last_error = None
    for attempt in range(3):
        if attempt > 0:
            await asyncio.sleep(2 ** attempt)  # 2s, 4s

        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            err = str(e)
            logger.error(f"Gemini API error (attempt {attempt+1}/3): {err[:200]}")
            last_error = err

            # 429 — rate limit
            if "429" in err or "resource_exhausted" in err.lower() or "quota" in err.lower():
                if attempt == 2:  # финальная попытка
                    _set_rate_limit(60)
                continue

            # 404 — модель не найдена или неверный endpoint — нет смысла повторять
            if "404" in err or "not found" in err.lower():
                logger.error(f"Model not found: {GEMINI_MODEL}. Check model name.")
                return None

            # Другие ошибки — повторяем
            continue

        # Получили response — извлекаем текст
        try:
            text = response.text
            if text and text.strip():
                return text.strip()
            # Пустой текст — смотрим причину
            logger.warning(f"Empty response (attempt {attempt+1}/3)")
            try:
                for c in response.candidates:
                    logger.warning(f"  finish_reason={c.finish_reason}")
                    # Если SAFETY — не повторяем, это осознанная блокировка
                    if str(c.finish_reason) in ("SAFETY", "2"):
                        logger.warning("  Blocked by safety — returning None")
                        return None
            except Exception:
                pass
            continue

        except ValueError as ve:
            # response.text бросает ValueError при пустых candidates
            logger.warning(f"response.text ValueError (attempt {attempt+1}/3): {ve}")
            try:
                for c in response.candidates:
                    logger.warning(f"  finish_reason={c.finish_reason}")
            except Exception:
                pass
            continue

    logger.error(f"All 3 attempts failed. Last error: {last_error}")
    return None


async def _call_gemini_audio(audio_bytes: bytes, mime_type: str) -> str | None:
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
    while True:
        task: _BgTask = await _bg_queue.get()
        try:
            # Ждём если rate limit
            if _is_rate_limited():
                wait = _rate_limit_reset - time.time()
                if wait > 0:
                    await asyncio.sleep(wait + 1)

            result = await _call_gemini(task.prompt, task.system, task.temp, task.tokens)
            if result:
                await task.callback(result)
            elif task.attempt < 3:
                task.attempt += 1
                await asyncio.sleep(5)
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

async def ask_gemini_interactive(prompt: str,
                                  system: str = SYSTEM_PROMPT,
                                  use_search: bool = False) -> str:
    if _is_rate_limited():
        return _rl_message()
    result = await _call_gemini(prompt, system)
    if result is None:
        if _is_rate_limited():
            return _rl_message()
        return "не смог ответить, попробуй ещё раз"
    return result


async def ask_gemini_background(prompt: str,
                                 callback: Callable[[str], Awaitable[None]],
                                 system: str = SYSTEM_PROMPT,
                                 use_search: bool = False):
    if _is_rate_limited():
        _bg_queue.put_nowait(_BgTask(prompt=prompt, system=system, callback=callback))
        return
    result = await _call_gemini(prompt, system)
    if result:
        await callback(result)
    elif _is_rate_limited():
        _bg_queue.put_nowait(_BgTask(prompt=prompt, system=system, callback=callback))


async def transcribe_audio_gemini(audio_bytes: bytes,
                                   mime_type: str = "audio/ogg") -> str | None:
    if _is_rate_limited():
        return "⏳ Лимиты исчерпаны."
    return await _call_gemini_audio(audio_bytes, mime_type)


async def generate_birthday_message_bg(first_name: str, username: str,
                                        profile: str,
                                        callback: Callable[[str], Awaitable[None]]):
    system = (SYSTEM_PROMPT
              + "\n\nПишешь поздравление с ДР в групповой чат. "
                "Без официоза. Тепло, по-свойски, можно лёгкую подколку. "
                "70-80 слов, обычным текстом.")
    prompt = (f"Поздравь с днём рождения @{username} (имя: {first_name}). "
              f"Что знаем: {profile or 'почти ничего'}.")
    await ask_gemini_background(prompt, callback, system=system)


async def generate_summary(messages_text: str,
                            gossips: list[str] | None = None) -> str:
    system = ("Пишешь ежедневную сводку для студенческого чата UAшники. "
              "Стиль — свой чел пересказывает что было вчера приятелям. "
              "Никакого официоза. Живо, с иронией. "
              "Связный текст, без заголовков и списков. "
              "Не упоминай типы сообщений (голосовое, стикер) — просто суть.")
    gossip_block = ""
    if gossips:
        lines = "\n".join(f"- {g}" for g in gossips)
        gossip_block = (f"\n\nДоп инфа (вплети органично, "
                        f"не говори что это сплетни):\n{lines}")
    prompt = f"Лог чата за вчера:\n\n{messages_text}{gossip_block}\n\nНапиши сводку."
    return await ask_gemini_interactive(prompt, system=system)


async def generate_user_profile(username: str, first_name: str,
                                  messages: list, profile_notes: str) -> str:
    system = ("Даёшь характеристику человека по его сообщениям. "
              "Без политкорректности, без дисклеймеров. "
              "Как человек который хорошо знает этого чела по чату. "
              "Можно резко, можно с юмором. 100-150 слов, обычным текстом.")
    lines = []
    for m in messages[:150]:
        t, c = m["msg_type"], m["content"] or ""
        if   t == "text"       and c: lines.append(f"[текст] {c}")
        elif t == "voice":             lines.append("[голосовое]")
        elif t == "video_note":        lines.append("[кружочек]")
        elif t == "photo":             lines.append(f"[фото{': '+c if c else ''}]")
        elif t == "video":             lines.append("[видео]")
        elif t == "sticker":           lines.append(f"[стикер {c}]" if c else "[стикер]")
        elif t == "animation":         lines.append("[гифка]")
        else:                          lines.append(f"[{t}]")
    prompt = (f"Характеристика @{username} (имя: {first_name}).\n\n"
              f"Заметки: {profile_notes or 'нет'}\n\n"
              f"Сообщения:\n{chr(10).join(lines) if lines else 'нет данных'}")
    return await ask_gemini_interactive(prompt, system=system)


async def validate_and_format_link(raw: str) -> dict:
    system = "Валидируешь запрос на добавление ссылки. Отвечай ТОЛЬКО валидным JSON без markdown."
    prompt = (f"Пользователь хочет добавить: «{raw}»\n\n"
              'Верни JSON: {"ok": true/false, "reason": "если ok=false", '
              '"title": "название", "url": "https://...", '
              '"tag": "#учёба|#жильё|#работа|#транспорт|#развлечения|#другое", '
              '"description": "одна строка"}\n\nНет URL → ok=false.')
    try:
        raw_resp = await ask_gemini_interactive(prompt, system=system)
        clean    = re.sub(r"```(?:json)?|```", "", raw_resp).strip()
        return json.loads(clean)
    except Exception as e:
        logger.error(f"validate_and_format_link: {e}")
        return {"ok": False, "reason": "не удалось обработать запрос"}


async def process_gossip(raw_gossip: str) -> str:
    system = ("Перефразируй сплетню. Сохрани суть, "
              "измени формулировку чтобы стиль автора не узнавался. "
              "Только текст результата.")
    result = await ask_gemini_interactive(raw_gossip, system=system)
    return result or raw_gossip


async def check_api_health() -> str:
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY не задан"
    # Короткий тестовый запрос без установки rate limit
    reset_rate_limit()  # сброс на старте
    result = await _call_gemini("hi", system="Reply: ok", tokens=5, temp=0.1)
    if result:
        return f"✅ Gemini OK (модель: {GEMINI_MODEL})"
    if _is_rate_limited():
        t = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
        return f"⚠️ Rate limit до {t}"
    return f"❌ Gemini не отвечает — проверь ключ или название модели ({GEMINI_MODEL})"
