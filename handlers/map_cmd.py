"""
/map — карта «СССР» в Университете Аликанте.
Фото + кликабельные ссылки.
"""

import os
from telegram import Update
from telegram.ext import ContextTypes

MAP_TEXT = (
    "🗺 *Карта UAшников — СССР в Аликанте*\n\n"
    "🔴 [СССР / Москва](https://maps.app.goo.gl/BPHrPkPq1jQCNPcz5) _— Cafetería Facultad Ciencias_\n"
    "🔴 [Киев](https://maps.app.goo.gl/q8sZ1M5pw1dqjgXt5)\n"
    "🔴 [Минск](https://maps.app.goo.gl/Y9GYbH61KTqW8dQQ7)\n"
    "\n"
    "─────────────────\n"
    "\n"
    "🔴 [Астана](https://maps.app.goo.gl/5LHULGUSwth5fe2G9) _— Club Social 1_\n"
    "🔴 [Ашхабад](https://maps.app.goo.gl/6MnRhPaCkUWGAhbn9) _— Туризм_\n"
    "🔴 [Баку](https://maps.app.goo.gl/KTa68ojah6oHfFvPA)\n"
    "🔴 [Бишкек](https://maps.app.goo.gl/qAubDN89UPWeTLkc6) _— Politécnica 4_\n"
    "🔴 [Варшава](https://maps.app.goo.gl/GRHzRTsygm86QGsS9)\n"
    "🔴 [Вильнюс](https://maps.app.goo.gl/oSfsowmmPSjnxZuz8) _— Aulario 2_\n"
    "🔴 [Душанбе](https://maps.app.goo.gl/ZbGRW1RcsFEpB8u58) _— Музей / Библиотека_\n"
    "🔴 [Ереван](https://maps.app.goo.gl/GNJzTi5gdoJuXFCp6)\n"
    "🔴 [Кишинёв](https://maps.app.goo.gl/LPFk9YhuYMzJF9hu7) _— Остановка 2_\n"
    "🔴 [Рига](https://maps.app.goo.gl/UZPooe6hvnXSiDtQA) _— Остановка 1_\n"
    "🔴 [София](https://maps.app.goo.gl/McKZ9qV4JXMBrHVz8) _— Главная Библиотека_\n"
    "🔴 [Таллин](https://maps.app.goo.gl/UHDkuxrCLayzqjq49)\n"
    "🔴 [Ташкент](https://maps.app.goo.gl/Bue2DLELt4s14tmS9) _— Aulario 1_\n"
    "🔴 [Тбилиси](https://maps.app.goo.gl/x9ZVxnReG3pkSPk96)\n"
    "🔴 [Хельсинки](https://maps.app.goo.gl/AFzpjcXBArEnWrm48)\n"
)

# Путь к картинке относительно корня проекта
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAP_IMAGE = os.path.join(_HERE, "..", "map.png")


async def cmd_map(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    try:
        with open(_MAP_IMAGE, "rb") as f:
            await msg.reply_photo(
                photo=f,
                caption=MAP_TEXT,
                parse_mode="Markdown")
    except FileNotFoundError:
        await msg.reply_text(MAP_TEXT, parse_mode="Markdown")
