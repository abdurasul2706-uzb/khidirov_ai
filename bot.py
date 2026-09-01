import asyncio
import logging
import os
import random
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command, CommandStart
from google import genai
from google.genai import types as genai_types


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("telegram-ai")


# ============================================================
# ENVIRONMENT
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

RAW_KEYS = os.getenv("GEMINI_API_KEY", "")

API_KEYS = [
    key.strip()
    for key in RAW_KEYS.split(",")
    if key.strip()
]

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)

# Reasoning:
# low    = tez
# medium = balans
# high   = maksimal reasoning
THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    "high"
)


# ============================================================
# VALIDATION
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN Render Environment Variables'da mavjud emas!"
    )

if not API_KEYS:
    raise RuntimeError(
        "GEMINI_API_KEY Render Environment Variables'da mavjud emas!"
    )


# ============================================================
# GEMINI CLIENTS
# ============================================================

clients = [
    genai.Client(api_key=key)
    for key in API_KEYS
]


# ============================================================
# AI PERSONALITY / SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
Siz Telegram ichidagi universal va yuqori darajadagi AI yordamchisiz.

Sizning asosiy maqsadingiz:
foydalanuvchiga imkon qadar aniq, foydali, aqlli, mantiqiy,
samimiy va amaliy yordam berish.

MUHIM QOIDALAR:

1. TIL
Foydalanuvchi qaysi tilda murojaat qilsa, asosan o'sha tilda javob bering.
O'zbek tilida lotin yoki kirill yozilsa, mos ravishda javob bering.
Ruscha savolga ruscha.
Inglizcha savolga inglizcha.
Aralash tilda yozilsa, foydalanuvchining asosiy tilini aniqlab,
shu tilda javob bering.

2. ANIQLIK
Bilmagan narsangizni uydirmang.
Noaniq ma'lumotni aniq fakt sifatida ko'rsatmang.
Agar savolda yetarli ma'lumot bo'lmasa, kerakli aniqlashtirishni so'rang.

3. MANTIQ
Murakkab savollarni bosqichma-bosqich tahlil qiling.
Matematika, mantiq, dasturlash va texnik savollarda
javobni tekshirishga harakat qiling.

4. JAVOB SIFATI
Javoblar:
- tushunarli
- mantiqiy
- tartibli
- foydali
- keraksiz takrorsiz
bo'lsin.

Kerak bo'lsa:
sarlavhalar,
ro'yxatlar,
jadval,
misollar,
kod bloklaridan foydalaning.

5. RASM
Rasm yuborilsa, uni diqqat bilan tahlil qiling.
Undagi matn, obyektlar, diagrammalar, grafiklar va kontekstni
imkon qadar tushuning.
Foydalanuvchi savol bergan bo'lsa, aynan shu savolga javob bering.

6. OVOZ
Ovozli xabar yuborilsa:
- undagi nutqni tushuning,
- savol yoki topshiriqni aniqlang,
- keyin foydalanuvchi tilida javob bering.

7. KOD
Dasturlash savollarida ishlaydigan, aniq va xavfsiz kod yozing.
Kod kerak bo'lsa, to'liq misol keltiring.
Xatolarni ham tushuntiring.

8. O'QITISH
Talaba yoki yangi o'rganuvchi savol bersa,
murakkab narsani sodda qilib tushuntiring.
Kerak bo'lsa oddiy misoldan boshlang.

9. SAMIMIYLIK
Foydalanuvchi bilan insoniy, hurmatli va samimiy muloqot qiling.
Lekin ortiqcha maqtov yoki sun'iy gaplardan qoching.

10. MUHIM
Sizning maqsadingiz shunchaki javob berish emas,
foydalanuvchining muammosini imkon qadar oxirigacha hal qilish.

Agar vazifa murakkab bo'lsa, javobni tartibli ravishda bering.
"""


# ============================================================
# USER MEMORY
# ============================================================

# Hozircha RAM'dagi vaqtinchalik xotira.
# Keyingi bosqichda PostgreSQL bilan doimiy xotiraga o'tamiz.

user_histories: dict[int, list[dict[str, str]]] = defaultdict(list)

MAX_HISTORY_MESSAGES = 12


def add_to_history(
    user_id: int,
    role: str,
    text: str,
) -> None:

    history = user_histories[user_id]

    history.append(
        {
            "role": role,
            "text": text,
        }
    )

    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]


def build_prompt(
    user_id: int,
    current_message: str,
) -> str:

    history = user_histories.get(user_id, [])

    if not history:
        return current_message

    previous_messages = []

    for item in history:
        role = item["role"]
        text = item["text"]

        if role == "user":
            previous_messages.append(
                f"Foydalanuvchi: {text}"
            )
        else:
            previous_messages.append(
                f"AI yordamchi: {text}"
            )

    history_text = "\n\n".join(previous_messages)

    return f"""
Quyida ushbu foydalanuvchi bilan oldingi suhbatning qisqa tarixi bor.

--- OLDINGI SUHBAT ---
{history_text}
--- SUHBAT OXIRI ---

Endi foydalanuvchining yangi xabari:

{current_message}

Yuqoridagi suhbat kontekstini hisobga olib, yangi xabarga javob bering.
"""


def reset_user_memory(user_id: int) -> None:
    user_histories.pop(user_id, None)


# ============================================================
# GEMINI REQUEST
# ============================================================

def generate_with_gemini(
    contents: Any,
):
    """
    Gemini'ga request yuboradi.

    Birinchi client xato qilsa,
    boshqa API client bilan qayta urinadi.
    """

    client_order = list(range(len(clients)))

    random.shuffle(client_order)

    last_error = None

    for index in client_order:

        client = clients[index]

        for attempt in range(3):

            try:

                logger.info(
                    "Gemini request | client=%s | attempt=%s",
                    index + 1,
                    attempt + 1,
                )

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        thinking_config=genai_types.ThinkingConfig(
                            thinking_level=THINKING_LEVEL
                        ),
                    ),
                )

                if response and response.text:
                    return response

                raise RuntimeError(
                    "Gemini bo'sh javob qaytardi."
                )

            except Exception as error:

                last_error = error

                logger.exception(
                    "Gemini error | client=%s | attempt=%s",
                    index + 1,
                    attempt + 1,
                )

                # Exponential backoff
                delay = 2 ** attempt

                time_to_wait = min(delay, 8)

                import time
                time.sleep(time_to_wait)

    raise RuntimeError(
        f"Gemini bilan bog'lanib bo'lmadi: {last_error}"
    )


# ============================================================
# TELEGRAM LONG MESSAGE SUPPORT
# ============================================================

TELEGRAM_MAX_LENGTH = 4000


def split_text(text: str, max_length: int = TELEGRAM_MAX_LENGTH):
    """
    Uzun Gemini javoblarini Telegram limitiga mos bo'ladi.
    Iloji boricha paragraf yoki qator bo'yicha ajratadi.
    """

    if len(text) <= max_length:
        return [text]

    chunks = []

    remaining = text.strip()

    while len(remaining) > max_length:

        cut = remaining.rfind(
            "\n",
            0,
            max_length,
        )

        if cut < max_length // 2:
            cut = remaining.rfind(
                " ",
                0,
                max_length,
            )

        if cut < max_length // 2:
            cut = max_length

        chunk = remaining[:cut].strip()

        if chunk:
            chunks.append(chunk)

        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


async def send_long_message(
    message: types.Message,
    text: str,
):

    chunks = split_text(text)

    for chunk in chunks:

        try:
            await message.answer(
                chunk,
                parse_mode=None,
            )

        except Exception:

            # Agar Telegram Markdown/format bilan bog'liq
            # muammo qilsa, plain text sifatida yuboriladi.
            await message.answer(
                str(chunk),
                parse_mode=None,
            )


# ============================================================
# BOT
# ============================================================

bot = Bot(
    token=BOT_TOKEN
)

dp = Dispatcher()


# ============================================================
# HEALTH CHECK SERVER
# ============================================================

class HealthCheckHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
        )
        self.end_headers()

        self.wfile.write(
            b"Telegram AI is alive."
        )

    def do_HEAD(self):

        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        return


def run_health_check_server():

    port = int(
        os.getenv("PORT", "10000")
    )

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthCheckHandler,
    )

    logger.info(
        "Health server started on port %s",
        port,
    )

    server.serve_forever()


# ============================================================
# /START
# ============================================================

@dp.message(CommandStart())
async def start_handler(
    message: types.Message,
):

    await message.answer(
        "Assalomu alaykum! 🤝\n\n"
        "Men sizning universal AI yordamchingizman. 🧠\n\n"
        "Menga:\n"
        "💬 matn\n"
        "🖼 rasm\n"
        "🎙 ovozli xabar\n"
        "yuborishingiz mumkin.\n\n"
        "Men o'zbek, rus, ingliz va boshqa tillarda "
        "muloqot qila olaman.\n\n"
        "Buyruqlar:\n"
        "/help — yordam\n"
        "/reset — suhbat xotirasini tozalash"
    )


# ============================================================
# /HELP
# ============================================================

@dp.message(Command("help"))
async def help_handler(
    message: types.Message,
):

    await message.answer(
        "🤖 AI yordamchidan quyidagicha foydalanishingiz mumkin:\n\n"
        "💬 Savol yozing — javob beraman.\n"
        "🖼 Rasm yuboring — tahlil qilaman.\n"
        "🎙 Ovoz yuboring — tinglab, javob beraman.\n\n"
        "🌍 O'zbek, rus, ingliz va boshqa tillar.\n"
        "🧠 Murakkab mantiqiy va texnik savollar.\n"
        "💻 Dasturlash va kod.\n"
        "📚 Ta'lim va tushuntirish.\n\n"
        "/reset — suhbat xotirasini tozalaydi."
    )


# ============================================================
# /RESET
# ============================================================

@dp.message(Command("reset"))
async def reset_handler(
    message: types.Message,
):

    reset_user_memory(
        message.from_user.id
    )

    await message.answer(
        "🧠 Suhbat xotirasi tozalandi.\n\n"
        "Yangi suhbatni boshlashimiz mumkin."
    )


# ============================================================
# TEXT HANDLER
# ============================================================

@dp.message(F.text)
async def text_handler(
    message: types.Message,
):

    user_id = message.from_user.id

    user_text = message.text.strip()

    if not user_text:
        return

    processing = await message.answer(
        "🧠 O'ylayapman..."
    )

    try:

        prompt = build_prompt(
            user_id,
            user_text,
        )

        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(
            None,
            lambda: generate_with_gemini(prompt),
        )

        answer = (
            response.text.strip()
            if response.text
            else "Kechirasiz, javob bo'sh qaytdi."
        )

        # History'ga faqat muvaffaqiyatli javobdan keyin qo'shamiz.
        add_to_history(
            user_id,
            "user",
            user_text,
        )

        add_to_history(
            user_id,
            "assistant",
            answer,
        )

        # Processing xabarini olib tashlaymiz.
        try:
            await processing.delete()
        except Exception:
            pass

        await send_long_message(
            message,
            answer,
        )

    except Exception as error:

        logger.exception(
            "Text handler error: %s",
            error,
        )

        try:
            await processing.edit_text(
                "⚠️ Hozircha javob olishda texnik muammo yuz berdi.\n\n"
                "Bir necha soniyadan keyin yana urinib ko'ring."
            )
        except Exception:
            await message.answer(
                "⚠️ Hozircha texnik muammo yuz berdi. "
                "Bir necha soniyadan keyin yana urinib ko'ring."
            )


# ============================================================
# PHOTO HANDLER
# ============================================================

@dp.message(F.photo)
async def photo_handler(
    message: types.Message,
):

    processing = await message.answer(
        "🖼 Rasmni sinchiklab tahlil qilyapman..."
    )

    try:

        photo = message.photo[-1]

        file_info = await bot.get_file(
            photo.file_id
        )

        downloaded_file = await bot.download_file(
            file_info.file_path
        )

        image_bytes = downloaded_file.read()

        prompt = (
            message.caption.strip()
            if message.caption
            else
            "Ushbu rasmni batafsil tahlil qiling. "
            "Rasmda nima borligini tushuntiring va "
            "agar unda matn, masala, diagramma yoki "
            "savol bo'lsa, uni ham tahlil qiling."
        )

        image_part = genai_types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )

        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(
            None,
            lambda: generate_with_gemini(
                [
                    prompt,
                    image_part,
                ]
            ),
        )

        answer = (
            response.text.strip()
            if response.text
            else "Rasmni tahlil qilib bo'lmadi."
        )

        try:
            await processing.delete()
        except Exception:
            pass

        await send_long_message(
            message,
            answer,
        )

    except Exception as error:

        logger.exception(
            "Photo handler error: %s",
            error,
        )

        try:
            await processing.edit_text(
                "⚠️ Rasmni tahlil qilishda texnik xatolik yuz berdi."
            )
        except Exception:
            await message.answer(
                "⚠️ Rasmni tahlil qilishda xatolik yuz berdi."
            )


# ============================================================
# VOICE HANDLER
# ============================================================

@dp.message(F.voice)
async def voice_handler(
    message: types.Message,
):

    processing = await message.answer(
        "🎙 Ovozli xabarni tinglayapman..."
    )

    try:

        voice = message.voice

        file_info = await bot.get_file(
            voice.file_id
        )

        downloaded_file = await bot.download_file(
            file_info.file_path
        )

        audio_bytes = downloaded_file.read()

        audio_part = genai_types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/ogg",
        )

        prompt = """
Ushbu ovozli xabarni diqqat bilan tinglang.

1. Undagi nutqni tushuning.
2. Foydalanuvchi nima so'rayotganini aniqlang.
3. Agar savol yoki topshiriq bo'lsa, uni bajaring.
4. Javobni foydalanuvchi gapirgan asosiy tilda bering.
5. Faqat transkripsiya berish bilan cheklanib qolmang.
"""

        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(
            None,
            lambda: generate_with_gemini(
                [
                    prompt,
                    audio_part,
                ]
            ),
        )

        answer = (
            response.text.strip()
            if response.text
            else "Ovozli xabarni tushunib bo'lmadi."
        )

        try:
            await processing.delete()
        except Exception:
            pass

        await send_long_message(
            message,
            answer,
        )

    except Exception as error:

        logger.exception(
            "Voice handler error: %s",
            error,
        )

        try:
            await processing.edit_text(
                "⚠️ Ovozli xabarni tahlil qilishda "
                "texnik xatolik yuz berdi."
            )
        except Exception:
            await message.answer(
                "⚠️ Ovozli xabarni tahlil qilishda xatolik yuz berdi."
            )


# ============================================================
# MAIN
# ============================================================

async def main():

    Thread(
        target=run_health_check_server,
        daemon=True,
    ).start()

    logger.info(
        "Telegram AI ishga tushmoqda..."
    )

    logger.info(
        "Model: %s",
        MODEL_NAME,
    )

    logger.info(
        "API clients: %s",
        len(clients),
    )

    logger.info(
        "Thinking level: %s",
        THINKING_LEVEL,
    )

    # Eski webhook/pending update'larni tozalash.
    await bot.delete_webhook(
        drop_pending_updates=True
    )

    await dp.start_polling(
        bot
    )


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:

        logger.info(
            "Bot to'xtatildi."
        )
