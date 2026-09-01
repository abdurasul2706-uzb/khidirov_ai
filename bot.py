import asyncio
import logging
import os
import random
import time
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


# ============================================================
# GEMINI MODELS
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
).strip()


# Fallback modellar.
#
# Agar kerak bo'lsa Render Environment Variables orqali:
#
# GEMINI_FALLBACK_MODELS=gemini-3.6-flash,gemini-3.5-flash
#
# ko'rinishida o'zgartirish mumkin.

RAW_FALLBACK_MODELS = os.getenv(
    "GEMINI_FALLBACK_MODELS",
    "gemini-3.6-flash,gemini-3.5-flash",
)

FALLBACK_MODELS = [
    model.strip()
    for model in RAW_FALLBACK_MODELS.split(",")
    if model.strip()
]


# ============================================================
# THINKING LEVEL
# ============================================================

# low    = tezroq
# medium = balans
# high   = chuqurroq reasoning

THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    "medium",
).strip().lower()


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

Foydalanuvchi qaysi tilda murojaat qilsa, asosan o'sha tilda
javob bering.

O'zbek tilida lotin yoki kirill yozilsa, mos ravishda javob bering.

Ruscha savolga ruscha.

Inglizcha savolga inglizcha.

Aralash tilda yozilsa, foydalanuvchining asosiy tilini aniqlab,
shu tilda javob bering.


2. ANIQLIK

Bilmagan narsangizni uydirmang.

Noaniq ma'lumotni aniq fakt sifatida ko'rsatmang.

Agar savolda yetarli ma'lumot bo'lmasa, kerakli aniqlashtirishni
so'rang.


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

- sarlavhalar
- ro'yxatlar
- jadval
- misollar
- kod bloklari

dan foydalaning.


5. RASM

Rasm yuborilsa, uni diqqat bilan tahlil qiling.

Undagi:

- matn
- obyektlar
- diagrammalar
- grafiklar
- hujjatlar
- masalalar

va boshqa muhim elementlarni imkon qadar tushuning.

Foydalanuvchi savol bergan bo'lsa, aynan shu savolga javob bering.


6. OVOZ

Ovozli xabar yuborilsa:

- undagi nutqni tushuning,
- foydalanuvchi nima so'rayotganini aniqlang,
- topshiriq bo'lsa bajaring,
- javobni foydalanuvchi tilida bering.

Faqat transkripsiya bilan cheklanib qolmang.


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


10. ASOSIY MAQSAD

Sizning maqsadingiz shunchaki javob berish emas.

Foydalanuvchining muammosini imkon qadar oxirigacha hal qilishga
harakat qiling.

Agar vazifa murakkab bo'lsa, javobni tartibli ravishda bering.
"""


# ============================================================
# USER MEMORY
# ============================================================

# Hozircha RAM'dagi vaqtinchalik xotira.
#
# Keyingi bosqichda PostgreSQL bilan doimiy memory qo'shamiz.

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
Quyida ushbu foydalanuvchi bilan oldingi suhbat tarixi bor.

--- OLDINGI SUHBAT ---

{history_text}

--- SUHBAT OXIRI ---

Endi foydalanuvchining yangi xabari:

{current_message}

Yuqoridagi suhbat kontekstini hisobga olib,
yangi xabarga javob bering.
"""


def reset_user_memory(
    user_id: int,
) -> None:

    user_histories.pop(
        user_id,
        None,
    )


# ============================================================
# GEMINI HELPERS
# ============================================================

def get_model_chain() -> list[str]:
    """
    Asosiy model + fallback modellarni tartibli ro'yxat qiladi.
    Takrorlangan modellar olib tashlanadi.
    """

    models = []

    all_models = [
        MODEL_NAME,
        *FALLBACK_MODELS,
    ]

    for model in all_models:

        if model and model not in models:
            models.append(model)

    return models


def is_temporary_gemini_error(
    error: Exception,
) -> bool:
    """
    Gemini vaqtinchalik server xatosini aniqlaydi.

    Masalan:

    503
    UNAVAILABLE
    Service Unavailable
    high demand
    overloaded
    """

    error_text = str(error).lower()

    temporary_signatures = [
        "503",
        "unavailable",
        "service unavailable",
        "currently experiencing high demand",
        "high demand",
        "temporarily unavailable",
        "overloaded",
        "internal server error",
        "deadline exceeded",
        "timeout",
    ]

    return any(
        signature in error_text
        for signature in temporary_signatures
    )


def generate_with_gemini(
    contents: Any,
):
    """
    Gemini request manager.

    Ishlash tartibi:

    1. Asosiy model
    2. 503 bo'lsa retry
    3. Hali ham ishlamasa fallback model
    4. API key'larni ham navbat bilan sinash
    5. Hammasi ishlamasa RuntimeError
    """

    model_chain = get_model_chain()

    client_order = list(
        range(len(clients))
    )

    # Client'larni tasodifiy tartibda ishlatamiz.
    random.shuffle(client_order)

    last_error = None

    logger.info(
        "Gemini model chain: %s",
        " -> ".join(model_chain),
    )

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model_position, model_name in enumerate(model_chain):

        is_fallback = model_position > 0

        if is_fallback:

            logger.warning(
                "Using fallback model | model=%s",
                model_name,
            )

        # ====================================================
        # API CLIENT LOOP
        # ====================================================

        for client_index in client_order:

            client = clients[client_index]

            # =================================================
            # RETRY LOOP
            # =================================================

            for attempt in range(3):

                try:

                    logger.info(
                        "Gemini request | "
                        "model=%s | client=%s | attempt=%s",
                        model_name,
                        client_index + 1,
                        attempt + 1,
                    )

                    # =========================================
                    # GEMINI REQUEST
                    # =========================================

                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            thinking_config=genai_types.ThinkingConfig(
                                thinking_level=THINKING_LEVEL
                            ),
                        ),
                    )

                    # =========================================
                    # EMPTY RESPONSE
                    # =========================================

                    if not response:

                        raise RuntimeError(
                            "Gemini response bo'sh qaytdi."
                        )

                    if not response.text:

                        raise RuntimeError(
                            "Gemini bo'sh matn qaytardi."
                        )

                    # =========================================
                    # SUCCESS
                    # =========================================

                    logger.info(
                        "Gemini success | "
                        "model=%s | client=%s",
                        model_name,
                        client_index + 1,
                    )

                    return response

                except Exception as error:

                    last_error = error

                    logger.error(
                        "Gemini error | "
                        "model=%s | client=%s | attempt=%s | error=%s",
                        model_name,
                        client_index + 1,
                        attempt + 1,
                        error,
                    )

                    # =========================================
                    # TEMPORARY ERROR
                    # =========================================

                    if is_temporary_gemini_error(error):

                        # 5 -> 10 -> 20 sekund
                        delay = min(
                            5 * (2 ** attempt),
                            20,
                        )

                        # Random jitter.
                        jitter = random.uniform(
                            0,
                            2,
                        )

                        total_delay = (
                            delay + jitter
                        )

                        logger.warning(
                            "Temporary Gemini error. "
                            "Retrying in %.1f seconds | "
                            "model=%s | attempt=%s",
                            total_delay,
                            model_name,
                            attempt + 1,
                        )

                        time.sleep(
                            total_delay
                        )

                        continue

                    # =========================================
                    # NON-TEMPORARY ERROR
                    # =========================================

                    logger.warning(
                        "Non-temporary Gemini error. "
                        "Trying next API client/model."
                    )

                    break

        # ====================================================
        # FALLBACK
        # ====================================================

        if model_position < len(model_chain) - 1:

            next_model = model_chain[
                model_position + 1
            ]

            logger.warning(
                "Model %s failed. "
                "Switching to %s",
                model_name,
                next_model,
            )

    # ========================================================
    # ALL FAILED
    # ========================================================

    raise RuntimeError(
        f"Gemini bilan bog'lanib bo'lmadi: {last_error}"
    )


# ============================================================
# TELEGRAM LONG MESSAGE SUPPORT
# ============================================================

TELEGRAM_MAX_LENGTH = 4000


def split_text(
    text: str,
    max_length: int = TELEGRAM_MAX_LENGTH,
) -> list[str]:
    """
    Uzun Gemini javoblarini Telegram limitiga moslaydi.

    Iloji boricha:

    1. paragraf
    2. yangi qator
    3. bo'sh joy

    bo'yicha ajratadi.
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_length:
        return [text]

    chunks = []

    remaining = text

    while len(remaining) > max_length:

        # Avval paragraf/yangi qator bo'yicha kesamiz.
        cut = remaining.rfind(
            "\n",
            0,
            max_length,
        )

        # Juda kichik chunk bo'lib qolsa,
        # oddiy space bo'yicha qidiramiz.
        if cut < max_length // 2:

            cut = remaining.rfind(
                " ",
                0,
                max_length,
            )

        # Hech qanday yaxshi joy topilmasa,
        # majburan limit bo'yicha kesamiz.
        if cut < max_length // 2:

            cut = max_length

        chunk = remaining[
            :cut
        ].strip()

        if chunk:
            chunks.append(chunk)

        remaining = remaining[
            cut:
        ].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


async def send_long_message(
    message: types.Message,
    text: str,
) -> None:

    chunks = split_text(text)

    if not chunks:

        await message.answer(
            "Kechirasiz, javob bo'sh qaytdi."
        )

        return

    for chunk in chunks:

        try:

            await message.answer(
                chunk,
                parse_mode=None,
            )

        except Exception:

            logger.exception(
                "Telegram message send error."
            )

            try:

                await message.answer(
                    str(chunk),
                    parse_mode=None,
                )

            except Exception:

                logger.exception(
                    "Telegram fallback message send error."
                )


# ============================================================
# BOT
# ============================================================

bot = Bot(
    token=BOT_TOKEN,
)

dp = Dispatcher()


# ============================================================
# HEALTH CHECK SERVER
# ============================================================

class HealthCheckHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8",
        )

        self.end_headers()

        self.wfile.write(
            b"Telegram AI is alive."
        )

    def do_HEAD(self):

        self.send_response(200)

        self.end_headers()

    def log_message(
        self,
        format,
        *args,
    ):
        return


def run_health_check_server():

    port = int(
        os.getenv(
            "PORT",
            "10000",
        )
    )

    server = HTTPServer(
        (
            "0.0.0.0",
            port,
        ),
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

@dp.message(
    CommandStart()
)
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

@dp.message(
    Command("help")
)
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

@dp.message(
    Command("reset")
)
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

@dp.message(
    F.text
)
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
            lambda: generate_with_gemini(
                prompt
            ),
        )

        answer = (
            response.text.strip()
            if response.text
            else "Kechirasiz, javob bo'sh qaytdi."
        )

        # ================================================
        # MEMORY
        # ================================================

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

        # ================================================
        # REMOVE PROCESSING MESSAGE
        # ================================================

        try:

            await processing.delete()

        except Exception:

            pass

        # ================================================
        # SEND ANSWER
        # ================================================

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

@dp.message(
    F.photo
)
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
            "Rasmda nima borligini tushuntiring. "
            "Agar unda matn, masala, diagramma, grafik, "
            "hujjat yoki savol bo'lsa, uni ham tahlil qiling."
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
                "⚠️ Rasmni tahlil qilishda texnik xatolik yuz berdi.\n\n"
                "Bir necha soniyadan keyin yana urinib ko'ring."
            )

        except Exception:

            await message.answer(
                "⚠️ Rasmni tahlil qilishda xatolik yuz berdi."
            )


# ============================================================
# VOICE HANDLER
# ============================================================

@dp.message(
    F.voice
)
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
                "texnik xatolik yuz berdi.\n\n"
                "Bir necha soniyadan keyin yana urinib ko'ring."
            )

        except Exception:

            await message.answer(
                "⚠️ Ovozli xabarni tahlil qilishda xatolik yuz berdi."
            )


# ============================================================
# MAIN
# ============================================================

async def main():

    # ========================================================
    # HEALTH SERVER
    # ========================================================

    Thread(
        target=run_health_check_server,
        daemon=True,
    ).start()

    # ========================================================
    # START LOGS
    # ========================================================

    logger.info(
        "Telegram AI ishga tushmoqda..."
    )

    logger.info(
        "Primary model: %s",
        MODEL_NAME,
    )

    logger.info(
        "Fallback models: %s",
        FALLBACK_MODELS,
    )

    logger.info(
        "API clients: %s",
        len(clients),
    )

    logger.info(
        "Thinking level: %s",
        THINKING_LEVEL,
    )

    # ========================================================
    # WEBHOOK CLEANUP
    # ========================================================

    await bot.delete_webhook(
        drop_pending_updates=True
    )

    # ========================================================
    # START POLLING
    # ========================================================

    await dp.start_polling(
        bot
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        logger.info(
            "Bot to'xtatildi."
        )
