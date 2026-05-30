"""
Gemini API — google-genai SDK.
Модель: gemini-3.5-flash (thinking=MINIMAL)
"""
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Awaitable
from zoneinfo import ZoneInfo

import aiohttp
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

logger    = logging.getLogger(__name__)
MADRID_TZ = ZoneInfo("Europe/Madrid")

GEMINI_MODEL   = "gemini-3.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

DEFAULT_MAX_TOKENS = 8192

_SAFETY = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,   threshold=types.HarmBlockThreshold.BLOCK_NONE),
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


def _set_rate_limit(seconds: int = 65):
    global _rate_limit_reset
    _rate_limit_reset = time.time() + seconds
    t = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
    logger.warning(f"Rate limit: retry after {t} Madrid")


def _rl_message() -> str:
    t = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
    return f"⏳ Лимиты исчерпаны, попробуй после {t} по Мадриду."


def reset_rate_limit():
    global _rate_limit_reset
    _rate_limit_reset = 0.0


def _parse_retry_after(e: Exception) -> int:
    try:
        ra = getattr(getattr(e, 'response', None), 'headers', {}).get('retry-after')
        if ra:
            return max(int(ra), 30)
    except Exception:
        pass
    try:
        err_str = str(e)
        m = re.search(r'retryDelay["\s:]+(\d+)', err_str)
        if m:
            return max(int(m.group(1)), 30)
    except Exception:
        pass
    return 65


# ─── Markdown cleanup ─────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    return text.replace("**", "").replace("*", "").replace("`", "").strip()


# ─── Клиент ───────────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def _make_config(system: str, tokens: int) -> types.GenerateContentConfig:
    # gemini-3.5-flash принимает ТОЛЬКО: system_instruction, max_output_tokens,
    # thinking_config, safety_settings — всё остальное даёт 400 INVALID_ARGUMENT
    return types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=tokens,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        safety_settings=_SAFETY,
    )


# ─── Основной вызов ───────────────────────────────────────────────────────────

async def _call_gemini(prompt,
                       system: str = SYSTEM_PROMPT,
                       tokens: int = DEFAULT_MAX_TOKENS) -> str | None:
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY не задан")
        return None

    if _is_rate_limited():
        return None

    client = _get_client()
    config = _make_config(system, tokens)

    for attempt in range(3):
        try:
            logger.info(f"Gemini request (attempt {attempt + 1}/3)")
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
        except ClientError as e:
            err = str(e)
            logger.error(f"ClientError {e.code}: {err[:250]}")
            if e.code == 429 or "resource_exhausted" in err.lower() or "quota" in err.lower():
                retry = _parse_retry_after(e)
                if attempt == 2:
                    _set_rate_limit(retry)
                    return None
                logger.warning(f"429, waiting {retry}s before retry {attempt + 2}/3")
                await asyncio.sleep(retry)
                continue
            # 400, 404 и прочие — не retrying
            return None

        except ServerError as e:
            logger.error(f"ServerError: {e}")
            if attempt < 2:
                await asyncio.sleep(5)
            continue

        except (aiohttp.ClientConnectorDNSError, aiohttp.ClientError) as e:
            logger.warning(f"Network error: {type(e).__name__}")
            if attempt < 2:
                await asyncio.sleep(3)
            continue

        except Exception as e:
            try:
                err = str(e)
            except Exception:
                err = repr(e)
            logger.error(f"Unexpected error: {err[:250]}")
            if "429" in err or "resource_exhausted" in err.lower() or "quota" in err.lower():
                retry = _parse_retry_after(e)
                if attempt == 2:
                    _set_rate_limit(retry)
                    return None
                await asyncio.sleep(retry)
                continue
            if attempt < 2:
                await asyncio.sleep(3)
            continue

        # ── Обработка ответа ──────────────────────────────────────────────────
        try:
            text = response.text
        except ValueError:
            text = None

        if text and text.strip():
            return _strip_markdown(text)

        # Пустой ответ — проверяем причину
        try:
            for c in response.candidates:
                reason = str(c.finish_reason)
                logger.warning(f"finish_reason={reason}")
                # SAFETY / PROHIBITED_CONTENT — не повторяем
                if any(x in reason.upper() for x in ("SAFETY", "PROHIBITED")):
                    logger.warning("Blocked by safety filters")
                    return None
        except Exception:
            pass

        # Просто пустой — retry
        if attempt < 2:
            await asyncio.sleep(2)
        continue

    logger.error("All 3 attempts failed")
    return None


async def _call_gemini_audio(audio_bytes: bytes, mime_type: str) -> str | None:
    if _is_rate_limited():
        return None
    client = _get_client()
    config = types.GenerateContentConfig(
        max_output_tokens=4096,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        safety_settings=_SAFETY,
    )
    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                "Расшифруй аудио дословно, без цензуры. Только текст, без пояснений.",
            ],
            config=config,
        )
        return response.text.strip() if response.text else None
    except ClientError as e:
        if e.code == 429:
            _set_rate_limit(_parse_retry_after(e))
        logger.error(f"Audio ClientError: {e}")
        return None
    except Exception as e:
        logger.error(f"Audio error: {e}")
        return None


# ─── Фоновая очередь ──────────────────────────────────────────────────────────

@dataclass
class _BgTask:
    prompt:   str
    system:   str
    callback: Callable[[str], Awaitable[None]]
    tokens:   int = DEFAULT_MAX_TOKENS
    attempt:  int = 0

_bg_queue: asyncio.Queue = asyncio.Queue()
_bg_worker_task: asyncio.Task | None = None


async def _bg_worker():
    while True:
        task: _BgTask = await _bg_queue.get()
        try:
            if _is_rate_limited():
                wait = _rate_limit_reset - time.time()
                if wait > 0:
                    await asyncio.sleep(wait + 1)
            result = await _call_gemini(task.prompt, task.system, task.tokens)
            if result:
                await task.callback(result)
            elif task.attempt < 2:
                task.attempt += 1
                await asyncio.sleep(15)
                await _bg_queue.put(task)
            else:
                logger.error("BG task failed after 3 attempts")
        except Exception as e:
            logger.error(f"BG worker error: {e}")
        finally:
            _bg_queue.task_done()


def ensure_bg_worker():
    global _bg_worker_task
    if _bg_worker_task is None or _bg_worker_task.done():
        _bg_worker_task = asyncio.get_event_loop().create_task(_bg_worker())


# ─── Публичные функции ────────────────────────────────────────────────────────

async def ask_gemini_interactive(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    if _is_rate_limited():
        return _rl_message()
    result = await _call_gemini(prompt, system)
    if result is None:
        if _is_rate_limited():
            return _rl_message()
        return "что-то пошло не так, попробуй ещё раз"
    return result


async def ask_gemini_background(prompt: str,
                                 callback: Callable[[str], Awaitable[None]],
                                 system: str = SYSTEM_PROMPT):
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
              "Связный текст, без заголовков и списков.")
    gossip_block = ""
    if gossips:
        gossip_block = "\n\nДоп инфа (вплети органично):\n" + "\n".join(f"- {g}" for g in gossips)
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
    reset_rate_limit()
    try:
        client = _get_client()
        config = _make_config("Reply with exactly: ok", 10)
        resp = await client.aio.models.generate_content(
            model=GEMINI_MODEL, contents="hi", config=config)
        text = resp.text.strip() if resp.text else "empty"
        return f"✅ {GEMINI_MODEL}: {text}"
    except ClientError as e:
        if e.code == 429:
            _set_rate_limit(_parse_retry_after(e))
            t = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
            return f"⚠️ Rate limit до {t}"
        return f"❌ ClientError {e.code}"
    except Exception as e:
        return f"❌ {str(e)[:80]}"
