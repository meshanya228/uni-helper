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
import httpx

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-1.5-flash-latest"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
MADRID_TZ      = ZoneInfo("Europe/Madrid")

# Safety — всё отключено
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

SYSTEM_PROMPT = """Ты — UniHelper, бот в закрытом Telegram-чате «UAшники» — русскоязычные студенты Университета Аликанте, Испания. Примерно 100 человек, все свои.

ХАРАКТЕР:
- Ты не ИИ-ассистент. Ты свой чувак в чате — с мозгами, с характером, без пафоса.
- Пиши как человек: без "Конечно!", "Отличный вопрос!", "Разумеется!", "Безусловно!".
- Не начинай с имени собеседника. Не благодари за вопрос.
- Маты, сленг, чёрный юмор — ок когда уместно. Не форси, но не бойся.
- Если ситуация серьёзная — отвечай серьёзно. Если стёб — стебись.
- Краткость. Не лей воду. Не пиши списками когда можно текстом.
- Эмодзи только если реально уместно.

ЗАПРЕЩЕНО:
- "Как ИИ, я..." / "Я языковая модель..."
- Извинения и дисклеймеры
- ** для выделения в обычных сообщениях
- Формальный тон там где его не ждут

Общайся на русском, если не попросят иначе."""

_rate_limit_reset: float = 0.0


class RateLimitError(Exception):
    def __init__(self, reset_at):
        self.reset_at = reset_at


@dataclass
class _BgTask:
    payload: dict
    callback: Callable[[str], Awaitable[None]]
    timeout: int = 90
    attempt: int = 0


_bg_queue: asyncio.Queue = asyncio.Queue()
_bg_worker_started = False


async def _raw_request(payload: dict, timeout: int = 90) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY не задан")
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            GEMINI_URL, json=payload,
            params={"key": GEMINI_API_KEY})
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 60))
        raise RateLimitError(time.time() + retry_after)
    # Логируем ошибки API подробно
    if resp.status_code != 200:
        logger.error(f"Gemini API error {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    return resp.json()


def _extract_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError) as e:
        logger.error(f"_extract_text failed: {e} | data keys: {list(data.keys())}")
        # Проверим есть ли блокировка
        if "promptFeedback" in data:
            logger.error(f"promptFeedback: {data['promptFeedback']}")
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
            else:
                logger.warning("BG worker: empty response from Gemini")
        except RateLimitError as e:
            _rate_limit_reset = e.reset_at
            task.attempt += 1
            if task.attempt < 5:
                await _bg_queue.put(task)
        except Exception as e:
            logger.error(f"BG worker error (attempt {task.attempt}): {e}")
        finally:
            _bg_queue.task_done()


def ensure_bg_worker():
    global _bg_worker_started
    if not _bg_worker_started:
        asyncio.get_event_loop().create_task(_bg_worker())
        _bg_worker_started = True



async def check_api_health() -> str:
    """Проверка доступности API - вызывается при старте."""
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY не задан"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"maxOutputTokens": 5},
    }
    try:
        data = await _raw_request(payload, timeout=15)
        text = _extract_text(data)
        return f"✅ Gemini API работает (модель: {GEMINI_MODEL})"
    except RateLimitError as e:
        return f"⚠️ Rate limit активен до {datetime.fromtimestamp(e.reset_at, tz=MADRID_TZ).strftime('%H:%M')}"
    except Exception as e:
        return f"❌ Gemini API ошибка: {e}"


def reset_rate_limit():
    """Сбросить rate limit вручную."""
    global _rate_limit_reset
    _rate_limit_reset = 0.0
    logger.info("Rate limit reset manually")

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
        # Правильный формат grounding для Gemini 2.5 Flash
        payload["tools"] = [{"google_search": {}}]
    return payload


def _enqueue_background(payload: dict, callback: Callable[[str], Awaitable[None]]):
    ensure_bg_worker()
    _bg_queue.put_nowait(_BgTask(payload=payload, callback=callback))


def _rl_message() -> str:
    dt = datetime.fromtimestamp(_rate_limit_reset, tz=MADRID_TZ).strftime('%H:%M')
    return f"⏳ Лимиты исчерпаны, попробуй после {dt} по Мадриду."


async def ask_gemini_interactive(prompt: str, system: str = SYSTEM_PROMPT,
                                  use_search: bool = False) -> str:
    """
    use_search=False по умолчанию — стабильнее.
    Передавай True только когда реально нужен интернет.
    """
    global _rate_limit_reset
    if time.time() < _rate_limit_reset:
        return _rl_message()
    payload = _build_payload(prompt, system, use_search=use_search)
    try:
        data = await _raw_request(payload)
        result = _extract_text(data)
        if not result:
            # Попробуем без search если был включён
            if use_search:
                logger.warning("Empty response with search, retrying without")
                payload2 = _build_payload(prompt, system, use_search=False)
                data2 = await _raw_request(payload2)
                result = _extract_text(data2)
        return result or "не смог ответить, попробуй ещё раз"
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
        data = await _raw_request(payload)
        text = _extract_text(data)
        if text:
            await callback(text)
        else:
            logger.warning("ask_gemini_background: empty response")
    except RateLimitError as e:
        _rate_limit_reset = e.reset_at
        _enqueue_background(payload, callback)
    except Exception as e:
        logger.error(f"Gemini background error: {e}")


async def transcribe_audio_gemini(audio_bytes: bytes,
                                   mime_type: str = "audio/ogg") -> str | None:
    global _rate_limit_reset
    if time.time() < _rate_limit_reset:
        return "⏳ Лимиты исчерпаны."
    payload = {
        "contents": [{"role": "user", "parts": [
            {"inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(audio_bytes).decode()}},
            {"text": "Расшифруй аудио дословно, без цензуры. Только текст, без пояснений."}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
        "safetySettings": SAFETY_SETTINGS,
    }
    try:
        data = await _raw_request(payload, timeout=120)
        return _extract_text(data) or None
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        return None


async def generate_birthday_message_bg(first_name: str, username: str,
                                        profile: str,
                                        callback: Callable[[str], Awaitable[None]]):
    system = (SYSTEM_PROMPT +
              "\n\nСейчас пишешь поздравление с ДР в групповой чат. "
              "Без официоза. Тепло, по-свойски, можно лёгкую подколку если есть за что. "
              "70-80 слов, без списков.")
    prompt = (f"Поздравь с днём рождения @{username} (имя: {first_name}). "
              f"Что знаем о человеке: {profile or 'почти ничего'}.")
    await ask_gemini_background(prompt, callback, system=system)


async def generate_summary(messages_text: str,
                            gossips: list[str] | None = None) -> str:
    system = (
        "Ты пишешь ежедневную сводку для студенческого чата UAшники. "
        "Стиль — как будто свой чел пересказывает что было вчера приятелям. "
        "Никакого официоза, никаких 'В течение дня участники обсуждали...'. "
        "Живо, с иронией, по-человечески. Связный текст, без заголовков и списков. "
        "Не упоминай типы сообщений (голосовое, стикер и т.д.) — просто передавай суть."
    )
    gossip_block = ""
    if gossips:
        lines = "\n".join(f"- {g}" for g in gossips)
        gossip_block = (
            f"\n\nДополнительная инфа (вплети органично в текст, "
            f"не говори что это сплетни и не выдели отдельным блоком):\n{lines}"
        )
    prompt = f"Лог чата за вчера:\n\n{messages_text}{gossip_block}\n\nНапиши сводку."
    return await ask_gemini_interactive(prompt, system=system, use_search=False)


async def generate_user_profile(username: str, first_name: str,
                                  messages: list, profile_notes: str) -> str:
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
        if mtype == "text" and content:
            msg_lines.append(f"[текст] {content}")
        elif mtype == "voice":
            msg_lines.append("[голосовое]")
        elif mtype == "video_note":
            msg_lines.append("[кружочек]")
        elif mtype == "photo":
            msg_lines.append(f"[фото{': ' + content if content else ''}]")
        elif mtype == "video":
            msg_lines.append("[видео]")
        elif mtype == "sticker":
            msg_lines.append(f"[стикер {content}]" if content else "[стикер]")
        elif mtype == "animation":
            msg_lines.append("[гифка]")
        else:
            msg_lines.append(f"[{mtype}]")

    msgs_str  = "\n".join(msg_lines) if msg_lines else "нет данных"
    notes_str = profile_notes or "нет"
    prompt = (
        f"Дай характеристику @{username} (имя: {first_name}).\n\n"
        f"Накопленные заметки: {notes_str}\n\n"
        f"Сообщения в чате:\n{msgs_str}"
    )
    return await ask_gemini_interactive(prompt, system=system, use_search=False)


async def validate_and_format_link(raw: str) -> dict:
    system = "Валидируешь запрос на добавление ссылки. Отвечай ТОЛЬКО валидным JSON без markdown."
    prompt = (
        f"Пользователь хочет добавить: «{raw}»\n\n"
        "Верни JSON:\n"
        '{"ok": true/false, "reason": "причина если ok=false", '
        '"title": "название", "url": "https://...", '
        '"tag": "#учёба|#жильё|#работа|#транспорт|#развлечения|#другое", '
        '"description": "одна строка"}\n\n'
        "Нет валидного URL → ok=false. Спам/реклама → ok=false."
    )
    try:
        raw_resp = await ask_gemini_interactive(prompt, system=system, use_search=False)
        clean = re.sub(r"```(?:json)?|```", "", raw_resp).strip()
        return json.loads(clean)
    except Exception as e:
        logger.error(f"validate_and_format_link error: {e}")
        return {"ok": False, "reason": "не удалось обработать запрос"}


async def process_gossip(raw_gossip: str) -> str:
    system = (
        "Перефразируй сплетню. Сохрани суть полностью, "
        "но измени формулировку так чтобы стиль автора не узнавался. "
        "Только текст результата, без пояснений."
    )
    result = await ask_gemini_interactive(raw_gossip, system=system, use_search=False)
    return result or raw_gossip
