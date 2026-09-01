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
from google.genai import errors as genai_errors


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

PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
).strip()

THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    "high",
).strip().lower()

ADMIN_USER_ID_RAW = os.getenv(
    "ADMIN_USER_ID",
    "",
).strip()


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

if not ADMIN_USER_ID_RAW:
    raise RuntimeError(
        "ADMIN_USER_ID Render Environment Variables'da mavjud emas!"
    )

try:
    ADMIN_USER_ID = int(ADMIN_USER_ID_RAW)
except ValueError:
    raise RuntimeError(
        "ADMIN_USER_ID noto'g'ri. Masalan: 970088832"
    )


# ============================================================
# MODEL FALLBACK
# ============================================================

# Tartib:
#
# 3.7 = asosiy, eng kuchli
# 3.6 = kuchli fallback
# 3.5 = yana bir fallback
# 3.5 Flash-Lite = tez va arzonroq fallback
#
# Rasm/audio ham kerak bo'lgani uchun fallback modellar
# multimodal bo'lishi muhim.

MODEL_FALLBACKS = [
    PRIMARY_MODEL,
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

# Duplicate model nomlarini olib tashlaymiz.
MODEL_FALLBACKS = list(
    dict.fromkeys(MODEL_FALLBACKS)
)


# ============================================================
# GEMINI CLIENTS
# ============================================================

clients = [
    genai.Client(api_key=key)
    for key in API_KEYS
]


# ============================================================
# AI PERSONALITY
# ============================================================

SYSTEM_INSTRUCTION = """
Siz Telegram ichidagi universal va yuqori darajadagi AI yordamchisiz.

Sizning asosiy maqsadingiz:
foydalanuvchiga imkon qadar aniq, foydali, aqlli,
mantiqiy, samimiy va amaliy yordam berish.

MUHIM QOIDALAR:

1. TIL

Foydalanuvchi qaysi tilda murojaat qilsa,
asosan o'sha tilda javob bering.

O'zbek tilida lotin yoki kirill yozilsa,
mos ravishda javob bering.

Ruscha savolga ruscha.
Inglizcha savolga inglizcha.

Aralash tilda yozilsa,
foydalanuvchining asosiy tilini aniqlab,
shu tilda javob bering.


2. ANIQLIK

Bilmagan narsangizni uydirmang.

Noaniq ma'lumotni aniq fakt sifatida ko'rsatmang.

Agar savolda yetarli ma'lumot bo'lmasa,
kerakli aniqlashtirishni so'rang.


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

Undagi matn, obyektlar, diagrammalar, grafiklar
va kontekstni imkon qadar tushuning.

Foydalanuvchi savol bergan bo'lsa,
aynan shu savolga javob bering.


6. OVOZ

Ovozli xabar yuborilsa:

- undagi nutqni tushuning
- savol yoki topshiriqni aniqlang
- keyin foydalanuvchi tilida javob bering


7. KOD

Dasturlash savollarida ishlaydigan,
aniq va xavfsiz kod yozing.

Kod kerak bo'lsa, to'liq misol keltiring.

Xatolarni ham tushuntiring.


8. O'QITISH

Talaba yoki yangi o'rganuvchi savol bersa,
murakkab narsani sodda qilib tushuntiring.


9. SAMIMIYLIK

Foydalanuvchi bilan insoniy,
hurmatli va samimiy muloqot qiling.

Lekin ortiqcha maqtov yoki sun'iy gaplardan qoching.


10. MUHIM

Sizning maqsadingiz shunchaki javob berish emas,
foydalanuvchining muammosini imkon qadar oxirigacha hal qilish.
"""


# ============================================================
# USER MEMORY
# ============================================================

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

    history = user_histories.get(
        user_id,
        [],
    )

    if not history:
        return current_message

    previous_messages = []

    for item in history:

        if item["role"] == "user":

            previous_messages.append(
                f"Foydalanuvchi: {item['text']}"
            )

        else:

            previous_messages.append(
                f"AI yordamchi: {item['text']}"
            )

    history_text = "\n\n".join(
        previous_messages
    )

    return f"""
Quyida ushbu foydalanuvchi bilan oldingi suhbat tarixi bor.

--- OLDINGI SUHBAT ---
{history_text}
--- SUHBAT OXIRI ---

Yangi xabar:

{current_message}

Oldingi suhbat kontekstini hisobga olib,
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
# USER STATISTICS
# ============================================================

users: dict[int, dict[str, Any]] = {}

user_message_counts: dict[int, int] = defaultdict(int)
user_photo_counts: dict[int, int] = defaultdict(int)
user_voice_counts: dict[int, int] = defaultdict(int)


def register_user(
    user: types.User,
) -> None:

    user_id = user.id

    now = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if user_id not in users:

        users[user_id] = {
            "id": user_id,
            "name": user.full_name or "Noma'lum",
            "username": user.username or "",
            "first_seen": now,
            "last_seen": now,
        }

        logger.info(
            "NEW USER | id=%s | name=%s | username=@%s",
            user_id,
            user.full_name,
            user.username or "-",
        )

    else:

        users[user_id]["name"] = (
            user.full_name or users[user_id]["name"]
        )

        users[user_id]["username"] = (
            user.username or users[user_id]["username"]
        )

        users[user_id]["last_seen"] = now


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def classify_gemini_error(
    error: Exception,
) -> str:

    text = str(error).lower()

    if "429" in text or "resource_exhausted" in text:
        return "quota"

    if "503" in text or "unavailable" in text:
        return "temporary"

    if "401" in text or "unauthenticated" in text:
        return "auth"

    if "403" in text or "permission" in text:
        return "permission"

    if "404" in text or "not found" in text:
        return "model"

    return "unknown"


# ============================================================
# GEMINI REQUEST
# ============================================================

def generate_with_gemini(
    contents: Any,
):

    last_error = None

    # API keylar ichida tasodifiy boshlaymiz.
    client_order = list(
        range(len(clients))
    )

    random.shuffle(
        client_order
    )

    for model_index, model_name in enumerate(
        MODEL_FALLBACKS
    ):

        logger.info(
            "Gemini model candidate %s/%s: %s",
            model_index + 1,
            len(MODEL_FALLBACKS),
            model_name,
        )

        for client_index in client_order:

            client = clients[client_index]

            # 503 uchun ko'pi bilan 2 retry.
            # 429 quota bo'lsa bekorga 3 marta kutmaymiz.
            max_attempts = 2

            for attempt in range(
                1,
                max_attempts + 1,
            ):

                try:

                    logger.info(
                        "Gemini request | model=%s | client=%s | attempt=%s",
                        model_name,
                        client_index + 1,
                        attempt,
                    )

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

                    if response and response.text:

                        logger.info(
                            "Gemini success | model=%s | client=%s",
                            model_name,
                            client_index + 1,
                        )

                        return response

                    raise RuntimeError(
                        "Gemini bo'sh javob qaytardi."
                    )

                except Exception as error:

                    last_error = error

                    error_type = classify_gemini_error(
                        error
                    )

                    logger.error(
                        "Gemini error | model=%s | client=%s | attempt=%s | type=%s | error=%s",
                        model_name,
                        client_index + 1,
                        attempt,
                        error_type,
                        error,
                    )

                    # ------------------------------------------------
                    # QUOTA
                    # ------------------------------------------------

                    if error_type == "quota":

                        logger.warning(
                            "Quota exceeded | model=%s | moving to next model/client",
                            model_name,
                        )

                        # Shu modelni qayta-qayta urmaymiz.
                        # Keyingi modelga o'tamiz.
                        break


                    # ------------------------------------------------
                    # AUTH
                    # ------------------------------------------------

                    if error_type in (
                        "auth",
                        "permission",
                    ):

                        logger.warning(
                            "API key/ruxsat muammosi | client=%s",
                            client_index + 1,
                        )

                        # Keyingi API key bo'lsa unga o'tamiz.
                        break


                    # ------------------------------------------------
                    # MODEL NOT FOUND
                    # ------------------------------------------------

                    if error_type == "model":

                        logger.warning(
                            "Model mavjud emas yoki project uchun ruxsat yo'q: %s",
                            model_name,
                        )

                        # Keyingi model.
                        break


                    # ------------------------------------------------
                    # 503 / TEMPORARY
                    # ------------------------------------------------

                    if error_type == "temporary":

                        if attempt < max_attempts:

                            delay = min(
                                2 ** attempt,
                                8,
                            )

                            # Ozgina random jitter.
                            delay += random.uniform(
                                0.2,
                                1.0,
                            )

                            logger.warning(
                                "Temporary Gemini error. %.1f sec kutamiz.",
                                delay,
                            )

                            time.sleep(
                                delay
                            )

                            continue

                        break


                    # ------------------------------------------------
                    # UNKNOWN
                    # ------------------------------------------------

                    if attempt < max_attempts:

                        time.sleep(2)

                    else:

                        break


    raise RuntimeError(
        f"Gemini bilan bog'lanib bo'lmadi: {last_error}"
    )


# ============================================================
# TELEGRAM MESSAGE SPLITTER
# ============================================================

TELEGRAM_MAX_LENGTH = 4000


def split_text(
    text: str,
    max_length: int = TELEGRAM_MAX_LENGTH,
):

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

        chunk = remaining[
            :cut
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        remaining = remaining[
            cut:
        ].strip()

    if remaining:

        chunks.append(
            remaining
        )

    return chunks


async def send_long_message(
    message: types.Message,
    text: str,
):

    for chunk in split_text(text):

        await message.answer(
            chunk,
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
# HEALTH SERVER
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
# ADMIN
# ============================================================

def is_admin(
    user_id: int,
) -> bool:

    return user_id == ADMIN_USER_ID


# ============================================================
# /START
# ============================================================

@dp.message(CommandStart())
async def start_handler(
    message: types.Message,
):

    if message.from_user:

        register_user(
            message.from_user
        )

    await message.answer(
        "Assalomu alaykum! 🤝\n\n"
        "Men sizning universal AI yordamchingizman. 🧠\n\n"
        "Menga:\n"
        "💬 matn\n"
        "🖼 rasm\n"
        "🎙 ovozli xabar\n"
        "yuborishingiz mumkin.\n\n"
        "O'zbek, rus, ingliz va boshqa tillarda "
        "muloqot qila olaman.\n\n"
        "Buyruqlar:\n"
        "/help — yordam\n"
        "/reset — suhbat xotirasini tozalash\n"
        "/id — Telegram ID"
    )


# ============================================================
# /HELP
# ============================================================

@dp.message(Command("help"))
async def help_handler(
    message: types.Message,
):

    if message.from_user:

        register_user(
            message.from_user
        )

    await message.answer(
        "🤖 AI yordamchidan foydalanish:\n\n"
        "💬 Savol yozing — javob beraman.\n"
        "🖼 Rasm yuboring — tahlil qilaman.\n"
        "🎙 Ovoz yuboring — tushunaman.\n\n"
        "🌍 O'zbek, rus, ingliz va boshqa tillar.\n"
        "🧠 Murakkab savollar.\n"
        "💻 Dasturlash va kod.\n"
        "📚 Ta'lim va tushuntirish.\n\n"
        "/reset — suhbat xotirasini tozalaydi.\n"
        "/id — Telegram ID."
    )


# ============================================================
# /ID
# ============================================================

@dp.message(Command("id"))
async def id_handler(
    message: types.Message,
):

    if not message.from_user:

        return

    register_user(
        message.from_user
    )

    await message.answer(
        "🆔 Sizning Telegram ID'ingiz:\n\n"
        f"{message.from_user.id}"
    )


# ============================================================
# /RESET
# ============================================================

@dp.message(Command("reset"))
async def reset_handler(
    message: types.Message,
):

    if message.from_user:

        register_user(
            message.from_user
        )

        reset_user_memory(
            message.from_user.id
        )

    await message.answer(
        "🧠 Suhbat xotirasi tozalandi.\n\n"
        "Yangi suhbatni boshlashimiz mumkin."
    )


# ============================================================
# /USERS
# ============================================================

@dp.message(Command("users"))
async def users_handler(
    message: types.Message,
):

    if not message.from_user:

        return

    if not is_admin(
        message.from_user.id
    ):

        await message.answer(
            "⛔ Bu buyruq faqat administrator uchun."
        )

        return

    total_users = len(users)

    total_messages = sum(
        user_message_counts.values()
    )

    total_photos = sum(
        user_photo_counts.values()
    )

    total_voice = sum(
        user_voice_counts.values()
    )

    lines = [
        "🔐 ADMIN PANEL",
        "",
        f"👥 Jami foydalanuvchilar: {total_users}",
        f"💬 Matn xabarlari: {total_messages}",
        f"🖼 Rasmlar: {total_photos}",
        f"🎙 Ovozli xabarlar: {total_voice}",
        "",
        "━━━━━━━━━━━━━━━━",
        "👥 FOYDALANUVCHILAR:",
        "",
    ]

    sorted_users = sorted(
        users.values(),
        key=lambda user: (
            user_message_counts.get(
                user["id"],
                0,
            )
            + user_photo_counts.get(
                user["id"],
                0,
            )
            + user_voice_counts.get(
                user["id"],
                0,
            )
        ),
        reverse=True,
    )

    for number, user in enumerate(
        sorted_users[:50],
        start=1,
    ):

        user_id = user["id"]

        username = user["username"]

        username_text = (
            f"@{username}"
            if username
            else "username yo'q"
        )

        lines.append(
            f"{number}. {user['name']}\n"
            f"   ├ ID: {user_id}\n"
            f"   ├ {username_text}\n"
            f"   ├ 💬 {user_message_counts.get(user_id, 0)} ta\n"
            f"   ├ 🖼 {user_photo_counts.get(user_id, 0)} ta\n"
            f"   └ 🎙 {user_voice_counts.get(user_id, 0)} ta"
        )

    if not users:

        lines.append(
            "Hozircha foydalanuvchilar yo'q."
        )

    await send_long_message(
        message,
        "\n".join(lines),
    )


# ============================================================
# TEXT
# ============================================================

@dp.message(F.text)
async def text_handler(
    message: types.Message,
):

    if not message.from_user:

        return

    user_id = message.from_user.id

    register_user(
        message.from_user
    )

    user_text = message.text.strip()

    if not user_text:

        return

    # Komandalar boshqa handlerlarda ishlaydi.
    if user_text.startswith("/"):

        return

    user_message_counts[
        user_id
    ] += 1

    logger.info(
        "TEXT | user_id=%s | name=%s | username=@%s | text=%s",
        user_id,
        message.from_user.full_name,
        message.from_user.username or "-",
        user_text[:200],
    )

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

        error_text = str(error).lower()

        if (
            "429" in error_text
            or "resource_exhausted" in error_text
            or "quota" in error_text
        ):

            user_error = (
                "⏳ AI xizmatining hozirgi quota limiti tugagan.\n\n"
                "Birozdan keyin yana urinib ko'ring."
            )

        elif (
            "503" in error_text
            or "unavailable" in error_text
        ):

            user_error = (
                "⏳ AI serveri vaqtincha band.\n\n"
                "Bir necha soniyadan keyin yana urinib ko'ring."
            )

        else:

            user_error = (
                "⚠️ Hozircha javob olishda texnik muammo yuz berdi.\n\n"
                "Bir necha soniyadan keyin yana urinib ko'ring."
            )

        try:

            await processing.edit_text(
                user_error
            )

        except Exception:

            await message.answer(
                user_error
            )


# ============================================================
# PHOTO
# ============================================================

@dp.message(F.photo)
async def photo_handler(
    message: types.Message,
):

    if not message.from_user:

        return

    user_id = message.from_user.id

    register_user(
        message.from_user
    )

    user_photo_counts[
        user_id
    ] += 1

    logger.info(
        "PHOTO | user_id=%s | name=%s | username=@%s",
        user_id,
        message.from_user.full_name,
        message.from_user.username or "-",
    )

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
            "Agar unda matn, masala, diagramma yoki "
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

        error_text = str(error).lower()

        if (
            "429" in error_text
            or "resource_exhausted" in error_text
        ):

            user_error = (
                "⏳ Rasm tahlili uchun AI quota limiti tugagan.\n\n"
                "Birozdan keyin yana urinib ko'ring."
            )

        else:

            user_error = (
                "⚠️ Rasmni tahlil qilishda texnik xatolik yuz berdi."
            )

        try:

            await processing.edit_text(
                user_error
            )

        except Exception:

            await message.answer(
                user_error
            )


# ============================================================
# VOICE
# ============================================================

@dp.message(F.voice)
async def voice_handler(
    message: types.Message,
):

    if not message.from_user:

        return

    user_id = message.from_user.id

    register_user(
        message.from_user
    )

    user_voice_counts[
        user_id
    ] += 1

    logger.info(
        "VOICE | user_id=%s | name=%s | username=@%s",
        user_id,
        message.from_user.full_name,
        message.from_user.username or "-",
    )

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
5. Faqat transkripsiya bilan cheklanib qolmang.
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

        error_text = str(error).lower()

        if (
            "429" in error_text
            or "resource_exhausted" in error_text
        ):

            user_error = (
                "⏳ Ovoz tahlili uchun AI quota limiti tugagan.\n\n"
                "Birozdan keyin yana urinib ko'ring."
            )

        else:

            user_error = (
                "⚠️ Ovozli xabarni tahlil qilishda "
                "texnik xatolik yuz berdi."
            )

        try:

            await processing.edit_text(
                user_error
            )

        except Exception:

            await message.answer(
                user_error
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
        "Primary model: %s",
        PRIMARY_MODEL,
    )

    logger.info(
        "Fallback models: %s",
        MODEL_FALLBACKS,
    )

    logger.info(
        "API clients: %s",
        len(clients),
    )

    logger.info(
        "Thinking level: %s",
        THINKING_LEVEL,
    )

    logger.info(
        "Admin ID: %s",
        ADMIN_USER_ID,
    )

    await bot.delete_webhook(
        drop_pending_updates=True
    )

    await dp.start_polling(
        bot
    )


# ============================================================
# RUN
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
