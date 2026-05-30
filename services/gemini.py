"""
Gemini API — официальный SDK google-genai.
Основная модель: gemini-3.5-flash (thinking=MINIMAL — без скрытых токенов рассуждения)
Fallback:        gemini-2.5-flash
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

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

logger    = logging.getLogger(__name__)
MADRID_TZ = ZoneInfo("Europe/Madrid")

# ─── Модели ───────────────────────────────────────────────────────────────────

GEMINI_MODEL_PRIMARY  = "gemini-3.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Токены вывода по умолчанию (для summary/profile/audio где нужно много).
# Токены вывода. Кириллица токенизируется в 2-3x хуже латиницы,
# поэтому нужен большой запас чтобы длинные ответы не обрывались.
DEFAULT_MAX_TOKENS = 8192

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

# ThinkingConfig для 3.5-flash — MINIMAL экономит тысячи токенов рассуждения,
# при этом модель остаётся умной (просто не думает так глубоко перед ответом)
_THINKING_MINIMAL = types.ThinkingConfig(thinking_level="MINIMAL")

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

# ─── Состояние моделей ─────────────────────────────────────────────────────────

@dataclass
class _ModelState:
    name: str
    # Поддерживает ли модель ThinkingConfig
    supports_thinking: bool = False
    # До какого времени заблокирована из-за 429
    blocked_until: float = 0.0
    # Последнее время запроса (для throttle)
    last_request: float = 0.0

_models: list[_ModelState] = [
    _ModelState(GEMINI_MODEL_PRIMARY,  supports_thinking=True),
    _ModelState(GEMINI_MODEL_FALLBACK, supports_thinking=False),
]

# Минимальный интервал между запросами к одной модели (сек)
_MIN_INTERVAL = 4.0

_model_locks: dict[str, asyncio.Lock] = {}


def _get_model_lock(name: str) -> asyncio.Lock:
    if name not in _model_locks:
        _model_locks[name] = asyncio.Lock()
    return _model_locks[name]


def _block_model(state: _ModelState, seconds: int):
    state.blocked_until = time.time() + seconds
    t = datetime.fromtimestamp(state.blocked_until, tz=MADRID_TZ).strftime('%H:%M')
    logger.warning(f"Model {state.name} blocked until {t} Madrid ({seconds}s)")


def _all_blocked_until() -> float:
    return min(m.blocked_until for m in _models)


def _is_all_blocked() -> bool:
    now = time.time()
    return all(now < m.blocked_until for m in _models)


def _rl_message() -> str:
    until = _all_blocked_until()
    t = datetime.fromtimestamp(until, tz=MADRID_TZ).strftime('%H:%M')
    return f"⏳ Лимиты исчерпаны, попробуй после {t} по Мадриду."


def reset_rate_limit():
    for m in _models:
        m.blocked_until = 0.0


def _parse_retry_after(e: Exception) -> int:
    try:
        response = getattr(e, 'response', None)
        if response is not None:
            headers = getattr(response, 'headers', {})
            ra = headers.get('retry-after') or headers.get('Retry-After')
            if ra:
                return max(int(ra), 30)
    except Exception:
        pass
    err_str = str(e)
    m = re.search(r'retryDelay["\s:]+(\d+)', err_str)
    if m:
        return max(int(m.group(1)), 30)
    return 65


# ─── Клиент ───────────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def _make_config(model_state: _ModelState,
                 system: str,
                 temp: float,
                 tokens: int) -> types.GenerateContentConfig:
    """
    Создаёт конфиг запроса.

    Для gemini-3.5-flash (thinking-модель):
      - temperature, top_p, top_k убраны — Google не рекомендует их для
        этой архитектуры, они конфликтуют с thinking и могут обрезать вывод
      - thinking_level=MINIMAL — отключает скрытое мышление, экономит TPM

    Для gemini-2.5-flash (fallback):
      - стандартные параметры генерации
    """
    if model_state.supports_thinking:
        # Thinking-модель: минимум параметров, только thinking + safety + токены
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=tokens,
            presence_penalty=0.65,
            frequency_penalty=0.50,
            thinking_config=_THINKING_MINIMAL,
            safety_settings=_SAFETY,
        )
    else:
        # Обычная модель: полный набор параметров
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=temp,
            top_p=0.95,
            top_k=40,
            presence_penalty=0.65,
            frequency_penalty=0.50,
            max_output_tokens=tokens,
            safety_settings=_SAFETY,
        )





def _strip_markdown(text: str) -> str:
    """Убирает markdown-разметку — бот пишет как человек, без форматирования."""
    # Заголовки: ## Текст → Текст (в начале строки)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = (
        text
        .replace("**", "")
        .replace("*", "")
        .replace("`", "")
    )
    return text.strip()


# ─── Один запрос к модели ─────────────────────────────────────────────────────

async def _call_model_once(client: genai.Client,
                            model_state: _ModelState,
                            prompt,
                            config: types.GenerateContentConfig) -> str | None:
    """
    Один запрос к конкретной модели.
    Возвращает текст или None (пустой ответ / safety block).
    Исключения пробрасываются наверх.
    """
    lock = _get_model_lock(model_state.name)
    async with lock:
        now = time.time()
        wait = model_state.last_request + _MIN_INTERVAL - now
        if wait > 0:
            logger.debug(f"Throttle {model_state.name}: sleeping {wait:.1f}s")
            await asyncio.sleep(wait)
        model_state.last_request = time.time()

    response = await client.aio.models.generate_content(
        model=model_state.name,
        contents=prompt,
        config=config,
    )

    try:
        text = response.text
        if text and text.strip():
            return _strip_markdown(text)
        # Пустой ответ — проверяем причину
        try:
            for c in response.candidates:
                reason = str(c.finish_reason)
                logger.warning(f"  {model_state.name}: finish_reason={reason}")
                if reason in ("SAFETY", "2"):
                    return None  # заблокировано safety — не retrying
        except Exception:
            pass
        return None
    except ValueError as e:
        logger.warning(f"  {model_state.name}: response.text ValueError: {e}")
        return None


# ─── Основной вызов с fallback ────────────────────────────────────────────────

async def _call_gemini(prompt,
                       system: str = SYSTEM_PROMPT,
                       temp: float = 0.9,
                       tokens: int = DEFAULT_MAX_TOKENS) -> str | None:
    """
    Вызов Gemini с автоматическим fallback.
    1. Пробуем gemini-3.5-flash (thinking=MINIMAL)
    2. Если 429 — автоматически переходим на gemini-2.5-flash
    3. Если оба недоступны — возвращаем None
    """
    if not GEMINI_API_KEY:
        return None

    client = _get_client()
    now = time.time()

    for model_state in _models:
        if now < model_state.blocked_until:
            remaining = model_state.blocked_until - now
            logger.info(f"Skipping {model_state.name} (blocked {remaining:.0f}s more)")
            continue

        config = _make_config(model_state, system, temp, tokens)
        logger.info(f"Trying model: {model_state.name}")

        try:
            result = await _call_model_once(client, model_state, prompt, config)
            if result is not None:
                return result
            # Пустой ответ — один retry через 2 сек, потом fallback
            logger.warning(f"{model_state.name}: empty response, retrying once...")
            await asyncio.sleep(2)
            result = await _call_model_once(client, model_state, prompt, config)
            if result is not None:
                return result
            logger.warning(f"{model_state.name}: empty again, trying next model")
            continue

        except ClientError as e:
            err = str(e)
            logger.error(f"{model_state.name} ClientError {e.code}: {err[:250]}")

            if e.code == 429 or "resource_exhausted" in err.lower() or "quota" in err.lower():
                retry_after = _parse_retry_after(e)
                _block_model(model_state, retry_after)
                logger.warning(f"Falling back from {model_state.name} after 429")
                continue

            if e.code == 404 or "not found" in err.lower():
                logger.error(f"Model {model_state.name} not found — blocking 24h")
                _block_model(model_state, 86400)
                continue

            # Другие 4xx — не retrying
            return None

        except ServerError as e:
            logger.error(f"{model_state.name} ServerError: {e}")
            await asyncio.sleep(2)
            continue

        except Exception as e:
            err = str(e)
            logger.error(f"{model_state.name} unexpected error: {err[:250]}")
            if "429" in err or "resource_exhausted" in err.lower() or "quota" in err.lower():
                _block_model(model_state, _parse_retry_after(e))
                continue
            if "404" in err or "not found" in err.lower():
                _block_model(model_state, 86400)
                continue
            await asyncio.sleep(2)
            continue

    logger.error("All models exhausted or blocked.")
    return None


async def _call_gemini_audio(audio_bytes: bytes, mime_type: str) -> str | None:
    client = _get_client()
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    text_part  = "Расшифруй аудио дословно, без цензуры. Только текст, без пояснений."
    for model_state in _models:
        if time.time() < model_state.blocked_until:
            continue
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
            safety_settings=_SAFETY,
        )
        try:
            response = await client.aio.models.generate_content(
                model=model_state.name,
                contents=[audio_part, text_part],
                config=config,
            )
            return response.text.strip() if response.text else None
        except ClientError as e:
            if e.code == 429:
                _block_model(model_state, _parse_retry_after(e))
                continue
            logger.error(f"Audio error ({model_state.name}): {e}")
            return None
        except Exception as e:
            logger.error(f"Audio error ({model_state.name}): {e}")
            return None
    return None


# ─── Фоновая очередь ──────────────────────────────────────────────────────────

@dataclass
class _BgTask:
    prompt:   str
    system:   str
    callback: Callable[[str], Awaitable[None]]
    temp:     float = 0.9
    tokens:   int   = DEFAULT_MAX_TOKENS
    attempt:  int   = 0

_bg_queue: asyncio.Queue = asyncio.Queue()
_bg_worker_task: asyncio.Task | None = None


async def _bg_worker():
    while True:
        task: _BgTask = await _bg_queue.get()
        try:
            if _is_all_blocked():
                wait = _all_blocked_until() - time.time()
                if wait > 0:
                    await asyncio.sleep(wait + 1)
            result = await _call_gemini(task.prompt, task.system, task.temp, task.tokens)
            if result:
                await task.callback(result)
            elif task.attempt < 2:
                task.attempt += 1
                await asyncio.sleep(15)
                await _bg_queue.put(task)
            else:
                logger.error("BG task failed after 3 attempts, dropping.")
        except Exception as e:
            logger.error(f"BG worker error: {e}")
        finally:
            _bg_queue.task_done()


def ensure_bg_worker():
    global _bg_worker_task
    if _bg_worker_task is None or _bg_worker_task.done():
        loop = asyncio.get_event_loop()
        _bg_worker_task = loop.create_task(_bg_worker())


# ─── Публичные функции ────────────────────────────────────────────────────────

async def ask_gemini_interactive(prompt: str,
                                  system: str = SYSTEM_PROMPT,
                                  use_search: bool = False) -> str:
    if _is_all_blocked():
        return _rl_message()
    result = await _call_gemini(prompt, system)
    if result is None:
        if _is_all_blocked():
            return _rl_message()
        return "что-то пошло не так, попробуй ещё раз"
    return result


async def ask_gemini_background(prompt: str,
                                 callback: Callable[[str], Awaitable[None]],
                                 system: str = SYSTEM_PROMPT,
                                 use_search: bool = False):
    if _is_all_blocked():
        _bg_queue.put_nowait(_BgTask(prompt=prompt, system=system, callback=callback))
        return
    result = await _call_gemini(prompt, system)
    if result:
        await callback(result)
    elif _is_all_blocked():
        _bg_queue.put_nowait(_BgTask(prompt=prompt, system=system, callback=callback))


async def transcribe_audio_gemini(audio_bytes: bytes,
                                   mime_type: str = "audio/ogg") -> str | None:
    if _is_all_blocked():
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
    # Сводка может быть длинной — даём больше токенов
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
    results = []
    client = _get_client()
    for model_state in _models:
        thinking_note = " [thinking=MINIMAL]" if model_state.supports_thinking else ""
        try:
            config = _make_config(model_state, "Reply with exactly: ok", 0.1, 10)
            resp = await client.aio.models.generate_content(
                model=model_state.name,
                contents="hi",
                config=config,
            )
            text = resp.text.strip() if resp.text else "empty"
            results.append(f"✅ {model_state.name}{thinking_note}: {text}")
        except ClientError as e:
            if e.code == 429:
                retry = _parse_retry_after(e)
                _block_model(model_state, retry)
                t = datetime.fromtimestamp(model_state.blocked_until,
                                           tz=MADRID_TZ).strftime('%H:%M')
                results.append(f"⚠️ {model_state.name}: rate limit до {t}")
            else:
                results.append(f"❌ {model_state.name}: {e.code}")
        except Exception as e:
            results.append(f"❌ {model_state.name}: {str(e)[:80]}")
    return "\n".join(results)
