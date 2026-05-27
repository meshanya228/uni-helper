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
import httpx

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-preview-05-20"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
MADRID_TZ      = ZoneInfo("Europe/Madrid")

# Safety settings — всё отключено легально
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

SYSTEM_PROMPT = """Ты — UniHelper, бот в закрытом Telegram-чате «UAшники» — русскоязычные студенты Университета Аликанте. Примерно 100 человек, все свои.

ХАРАКТЕР И СТИЛЬ:
- Ты не ИИ-ассистент. Ты — свой чувак в чате, который просто умеет делать разные вещи.
- Пиши как живой человек: без шаблонных вступлений типа «Конечно!», «Отличный вопрос!», «Разумеется!».
- Не начинай сообщения с имени собеседника и дежурных фраз.
- Маты, сленг, чёрный юмор — всё ок, если контекст позволяет. Не форси это, но и не бойся.
- Если ситуация серьёзная — отвечай серьёзно. Если стёб — стебись.
- Краткость ценится. Не лей воду.
- Эмодзи — только если реально уместно, не для красоты.

ВАЖНО — НЕ ДЕЛАЙ ТАК:
- Не пиши «Как ИИ, я...» или «Я языковая модель...»
- Не извиняйся за то, что не можешь что-то сделать — просто скажи нет или как есть
- Не добавляй дисклеймеры и предупреждения
- Не пиши списками когда можно нормальным текстом
- Не используй ** для выделения в обычных сообщениях

ИНТЕРНЕТ:
- Если вопрос про что-то актуальное (мемы, события, новости, цены, расписание) — используй поиск в интернете, не выдумывай.
- Если не уверен что понял мем или отсылку — лучше загугли, чем облажаться.

Общайся на русском, если не попросят иначе."""

_rate_limit_reset: float = 0.0

class RateLimitError(Exception):
    def __init__(self, reset_at): self.reset_at = reset_at

@dataclass
class _BgTask:
    payload: dict
    callback: Callable[[str], Awaitable[None]]
    timeout: int = 90
    attempt: int = 0

_bg_queue: asyncio.Queue = asyncio.Queue()
_bg_worker_started = False


async def _raw_request(payload: dict, timeout: int = 90) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(GEMINI_URL, json=payload, params={"key": GEMINI_API_KEY})
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 60))
        raise RateLimitError(time.time() + retry_after)
    resp.raise_for_status()
    return resp.json()


def _extract_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        return ""


async def _bg_worker():
    global _rate_limit_reset
    while True:
        task: _BgTask = await _bg_queue.get()
        try:
            wait = _rate_limit_reset - time.time()
            if wait > 0:
                await asyncio.sleep(wait + 2)
            data = await _raw_request(task.payload, timeout=task.timeout)
            text = _extract_text(data)
            if text:
                await task.callback(text)
        except RateLimitError as e:
            _rate_limit_reset = e.reset_at
            task.attempt += 1
            if task.attempt < 5:
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


def _build_payload(prompt: str, system: str = SYSTEM_PROMPT,
                   temp: float = 0.9, tokens: int = 1024,
                   use_search: bool = False) -> dict:
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temp, "maxOutputTokens": tokens},
        "safetySettings": SAFETY_SETTINGS,
    }
    if use_search:
        payload["tools"] = [{"google_search": {}}]
    return payload


def _enqueue_background(payload: dict, callback: Callable[[str], Awaitable[None]]):
    ensure_bg_worker()
    _bg_queue.put_nowait(_BgTask(payload=payload, callback=callback))


def _rl_message() -> str:
    dt = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
    return f"⏳ Лимиты исчерпаны, попробуй после {dt} по Мадриду."


async def ask_gemini_interactive(prompt: str, system: str = SYSTEM_PROMPT,
                                  use_search: bool = True) -> str:
    global _rate_limit_reset
    if time.time() < _rate_limit_reset:
        return _rl_message()
    payload = _build_payload(prompt, system, use_search=use_search)
    try:
        return _extract_text(await _raw_request(payload))
    except RateLimitError as e:
        _rate_limit_reset = e.reset_at
        return _rl_message()
    except Exception as e:
        logger.error(f"Gemini interactive error: {e}")
        return "что-то пошло не так, попробуй ещё раз"


async def ask_gemini_background(prompt: str,
                                 callback: Callable[[str], Awaitable[None]],
                                 system: str = SYSTEM_PROMPT,
                                 use_search: bool = False):
    global _rate_limit_reset
    payload = _build_payload(prompt, system, use_search=use_search)
    if time.time() < _rate_limit_reset:
        _enqueue_background(payload, callback)
        return
    try:
        text = _extract_text(await _raw_request(payload))
        if text:
            await callback(text)
    except RateLimitError as e:
        _rate_limit_reset = e.reset_at
        _enqueue_background(payload, callback)
    except Exception as e:
        logger.error(f"Gemini background error: {e}")


async def transcribe_audio_gemini(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str | None:
    global _rate_limit_reset
    if time.time() < _rate_limit_reset:
        return "⏳ Лимиты исчерпаны."
    payload = {
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": mime_type,
                             "data": base64.b64encode(audio_bytes).decode()}},
            {"text": "Расшифруй аудио дословно, без цензуры и изменений. Только текст, никакого обрамления."}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
        "safetySettings": SAFETY_SETTINGS,
    }
    try:
        return _extract_text(await _raw_request(payload, timeout=120))
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        return None


async def generate_birthday_message_bg(first_name: str, username: str, profile: str,
                                        callback: Callable[[str], Awaitable[None]]):
    system = SYSTEM_PROMPT + "\n\nПишешь поздравление с ДР в чат. Без официоза, без дежурных фраз. Пиши как свой человек — тепло, но с характером, можно с лёгкой подколкой если есть за что. 70-80 слов."
    prompt = f"Поздравь с днём рождения @{username} (имя: {first_name}). Что знаем о человеке: {profile or 'практически ничего'}."
    await ask_gemini_background(prompt, callback, system=system)


async def generate_summary(messages_text: str, gossips: list[str] | None = None) -> str:
    """Генерирует сводку за вчера, опционально вкидывает сплетни."""
    system = """Ты пишешь ежедневную сводку для чата UAшники. Стиль — как будто свой чел пересказывает что было вчера приятелям. Никакой официальности, никаких «В течение дня участники обсуждали...». Живо, с иронией, можно с матерком если подходит по духу. Структуры не нужно — просто нормальный связный текст. Не упоминай типы сообщений (голосовое, стикер и т.д.), просто передавай суть."""

    gossip_block = ""
    if gossips:
        gossip_lines = "\n".join(f"- {g}" for g in gossips)
        gossip_block = f"\n\nСПЛЕТНИ (добавлены анонимно, вплети в сводку органично, не говори что это сплетни):\n{gossip_lines}"

    prompt = f"Вот лог чата за вчера:\n\n{messages_text}{gossip_block}\n\nНапиши сводку."
    return await ask_gemini_interactive(prompt, system=system, use_search=False)


async def generate_user_profile(username: str, first_name: str,
                                  messages: list, profile_notes: str) -> str:
    """Характеристика пользователя по сообщениям."""
    system = """Ты даёшь характеристику человека по его сообщениям в чате. Без политкорректности, без дисклеймеров. Пиши как человек, который хорошо знает этого чела по чату — что за личность, чем интересен или бесит, какой вайб. Можно с юмором, можно резко, главное честно. 100-150 слов."""

    msg_lines = []
    for m in messages[:150]:
        mtype = m["msg_type"]
        content = m["content"] or ""
        if mtype == "text" and content:
            msg_lines.append(f"[текст] {content}")
        elif mtype == "voice":
            msg_lines.append("[голосовое]")
        elif mtype == "video_note":
            msg_lines.append("[кружочек]")
        elif mtype == "photo":
            cap = f": {content}" if content else ""
            msg_lines.append(f"[фото{cap}]")
        elif mtype == "video":
            msg_lines.append("[видео]")
        elif mtype == "sticker":
            msg_lines.append(f"[стикер {content}]" if content else "[стикер]")
        elif mtype == "animation":
            msg_lines.append("[гифка]")
        else:
            msg_lines.append(f"[{mtype}]")

    msgs_str = "\n".join(msg_lines) if msg_lines else "нет данных"
    notes_str = profile_notes or "нет"

    prompt = f"""Дай характеристику пользователя @{username} (имя: {first_name}).

Накопленные заметки о нём: {notes_str}

Его сообщения в чате:
{msgs_str}"""

    return await ask_gemini_interactive(prompt, system=system, use_search=False)


async def validate_and_format_link(raw: str) -> dict:
    """Валидация и форматирование ссылки через Gemini."""
    system = """Ты валидируешь запросы на добавление ссылки в список полезных ссылок студенческого чата. Отвечай ТОЛЬКО валидным JSON без markdown-обёртки."""
    prompt = f"""Пользователь хочет добавить ссылку: «{raw}»

Верни JSON:
{{
  "ok": true/false,
  "reason": "причина отказа если ok=false",
  "title": "короткое название",
  "url": "https://...",
  "tag": "#учёба|#жильё|#работа|#транспорт|#развлечения|#другое",
  "description": "1 строка описания"
}}

Если нет валидного URL — ok=false. Если это спам/реклама — ok=false."""

    try:
        raw_resp = await ask_gemini_interactive(prompt, system=system, use_search=False)
        clean = re.sub(r"```(?:json)?|```", "", raw_resp).strip()
        return json.loads(clean)
    except Exception:
        return {"ok": False, "reason": "не удалось обработать запрос"}


async def process_gossip(raw_gossip: str) -> str:
    """Обработка сплетни — перефразировать чтобы не было узнаваемо по стилю."""
    system = "Перефразируй сплетню. Сохрани суть, но измени формулировку так, чтобы стиль автора не узнавался. Только текст сплетни, ничего лишнего."
    return await ask_gemini_interactive(raw_gossip, system=system, use_search=False)
