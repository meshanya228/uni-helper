"""
/map — отправляет карту USSR/Университет Аликанте с кликабельными ссылками.
"""

import os
from telegram import Update
from telegram.ext import ContextTypes

# Порядок: СССР/Москва первая, Киев вторая, Минск третья, остальные по алфавиту
MAP_TEXT = """🗺 *Карта UAшников — СССР в Аликанте*

🔴 [СССР / Москва](https://maps.app.goo.gl/BPHrPkPq1jQCNPcz5) _(Cafetería Facultad Ciencias)_
🔴 [Киев](https://maps.app.goo.gl/q8sZ1M5pw1dqjgXt5)
🔴 [Минск](https://maps.app.goo.gl/Y9GYbH61KTqW8dQQ7)

─────────────────
🔴 [Астана](https://maps.app.goo.gl/5LHULGUSwth5fe2G9) _(Club Social 1)_
🔴 [Ашхабад](https://maps.app.goo.gl/6MnRhPaCkUWGAhbn9) _(Туризм)_
🔴 [Баку](https://maps.app.goo.gl/KTa68ojah6oHfFvPA)
🔴 [Бишкек](https://maps.app.goo.gl/qAubDN89UPWeTLkc6) _(Politécnica 4)_
🔴 [Варшава](https://maps.app.goo.gl/GRHzRTsygm86QGsS9)
🔴 [Вильнюс](https://maps.app.goo.gl/oSfsowmmPSjnxZuz8) _(Aulario 2)_
🔴 [Душанбе](https://maps.app.goo.gl/ZbGRW1RcsFEpB8u58) _(Музей / Библиотека)_
🔴 [Ереван](https://maps.app.goo.gl/GNJzTi5gdoJuXFCp6)
🔴 [Кишинёв](https://maps.app.goo.gl/LPFk9YhuYMzJF9hu7) _(Остановка 2)_
🔴 [Рига](https://maps.app.goo.gl/UZPooe6hvnXSiDtQA) _(Остановка 1)_
🔴 [София](https://maps.app.goo.gl/McKZ9qV4JXMBrHVz8) _(Главная Библиотека)_
🔴 [Таллин](https://maps.app.goo.gl/UHDkuxrCLayzqjq49)
🔴 [Ташкент](https://maps.app.goo.gl/Bue2DLELt4s14tmS9) _(Aulario 1)_
🔴 [Тбилиси](https://maps.app.goo.gl/x9ZVxnReG3pkSPk96)
🔴 [Хельсинки](https://maps.app.goo.gl/AFzpjcXBArEnWrm48)"""

# Путь к картинке — лежит рядом с кодом
_MAP_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "map.png")


async def cmd_map(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    try:
        with open(_MAP_IMAGE_PATH, "rb") as f:
            await msg.reply_photo(
                photo=f,
                caption=MAP_TEXT,
                parse_mode="Markdown"
            )
    except FileNotFoundError:
        # Фоллбэк — только текст если картинка не найдена
        await msg.reply_text(MAP_TEXT, parse_mode="Markdown")
