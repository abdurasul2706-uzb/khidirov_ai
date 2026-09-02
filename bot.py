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

from openai import OpenAI


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

BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    "",
).strip()

RAW_KEYS = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

API_KEYS = [
    key.strip()
    for key in RAW_KEYS.split(",")
    if key.strip()
]

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
).strip()

THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    "medium",
).strip().lower()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
).strip()

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
    ADMIN_USER_ID = int(
        ADMIN_USER_ID_RAW
    )
except ValueError:
    raise RuntimeError(
        "ADMIN_USER_ID noto'g'ri. Masalan: 970088832"
    )


# ============================================================
# GEMINI CLIENTS
# ============================================================

clients = [
    genai.Client(api_key=key)
    for key in API_KEYS
]


# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = (
    OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    if GROQ_API_KEY
    else None
)


# ============================================================
# AI SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
Siz Telegram ichidagi universal va yuqori darajadagi AI yordamchisiz.

Sizning asosiy maqsadingiz foydalanuvchining muammosini
imkon qadar oxirigacha hal qilish.

JAVOB SIFATI ENG MUHIM.


1. TIL

Foydalanuvchi qaysi tilda yozsa, o'sha tilda javob bering.

O'zbek lotin → o'zbek lotin.
O'zbek kirill → o'zbek kirill.
Ruscha → ruscha.
Inglizcha → inglizcha.

Aralash tilda yozilsa, asosiy tilni aniqlang.


2. MANTIQ VA REASONING

Murakkab savollarni shoshmasdan tahlil qiling.

Matematika, mantiq, dasturlash, texnika va murakkab
muammolarda javobni ichki reasoning orqali tekshiring.

Foydalanuvchiga keraksiz ichki reasoning jarayonini
ochib bermang.

Faqat foydali xulosa va tushuntirishni bering.


3. ANIQLIK

Bilmagan narsangizni uydirmang.

Noaniq ma'lumotni fakt sifatida ko'rsatmang.

Yetarli ma'lumot bo'lmasa, zarur bo'lsa aniqlashtiruvchi
savol bering.


4. JAVOB USLUBI

Javoblar:

- aniq
- aqlli
- tabiiy
- tushunarli
- amaliy
- tartibli

bo'lsin.

Keraksiz uzunlikdan qoching.

Murakkab masalalarda esa yetarlicha batafsil tushuntiring.


5. KOD

Dasturlash savollarida ishlaydigan kod yozing.

Xato bo'lsa:

- sababini aniqlang
- qaysi joy xato ekanini ayting
- to'g'ri variantni ko'rsating

Foydalanuvchi kod yuborsa, kodni diqqat bilan tahlil qiling.


6. RASM

Rasm yuborilsa:

- rasmni diqqat bilan ko'ring
- matnni o'qing
- obyektlarni aniqlang
- diagramma/grafikni tushuning
- masala yoki savol bo'lsa yeching

Foydalanuvchi caption yozgan bo'lsa,
aynan shu topshiriqqa e'tibor bering.


7. OVOZ

Ovozli xabar yuborilsa:

- nutq mazmunini tushuning
- foydalanuvchining topshirig'ini aniqlang
- faqat transkripsiya bilan cheklanmay,
  imkon qadar topshiriqni bajaring
- javobni foydalanuvchi tilida bering


8. SUHBAT KONTEKSTI

Oldingi suhbatdagi ma'lumotlardan foydalaning.

Foydalanuvchi "u", "bu", "avvalgi", "o'sha" kabi
iboralarni ishlatsa, oldingi kontekstni hisobga oling.


9. SAMIMIYLIK

Insoniy, tabiiy va hurmatli muloqot qiling.

Sun'iy maqtovlarni ko'paytirmang.


10. MUAMMONI HAL QILISH

Shunchaki javob berish emas,
foydalanuvchiga amalda yordam berish asosiy maqsad.


11. TELEGRAM FORMAT

Telegramda o'qishga qulay formatdan foydalaning:

- qisqa sarlavhalar
- punktlar
- raqamlangan ro'yxatlar
- kod bloklari

Kerak bo'lmasa haddan tashqari formatlamang.
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
OLDINGI SUHBAT KONTEKSTI:

{history_text}

---

YANGI FOYDALANUVCHI XABARI:

{current_message}

---

Oldingi suhbatni hisobga olib,
yangi xabarga tabiiy va foydali javob bering.
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
            user.full_name
            or users[user_id]["name"]
        )

        users[user_id]["username"] = (
            user.username
            or users[user_id]["username"]
        )

        users[user_id]["last_seen"] = now


# ============================================================
# ERROR TYPE
# ============================================================

def get_error_type(
    error: Exception,
) -> str:

    text = str(error).lower()

    if (
        "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "rate limit" in text
        or "rate_limit" in text
    ):
        return "quota"

    if (
        "503" in text
        or "unavailable" in text
        or "service unavailable" in text
        or "overloaded" in text
    ):
        return "temporary"

    if (
        "401" in text
        or "unauthenticated" in text
        or "invalid api key" in text
        or "invalid_api_key" in text
    ):
        return "auth"

    if (
        "403" in text
        or "permission denied" in text
        or "forbidden" in text
    ):
        return "permission"

    if (
        "404" in text
        or "not found" in text
    ):
        return "model"

    return "unknown"


# ============================================================
# GEMINI REQUEST
# ============================================================

def generate_with_gemini(
    contents: Any,
):

    last_error = None

    client_order = list(
        range(len(clients))
    )

    random.shuffle(
        client_order
    )

    for client_index in client_order:

        client = clients[client_index]

        for attempt in range(1, 4):

            try:

                logger.info(
                    "Gemini request | model=%s | client=%s | attempt=%s",
                    MODEL_NAME,
                    client_index + 1,
                    attempt,
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

                    logger.info(
                        "Gemini success | model=%s | client=%s",
                        MODEL_NAME,
                        client_index + 1,
                    )

                    return response

                raise RuntimeError(
                    "Gemini bo'sh javob qaytardi."
                )

            except Exception as error:

                last_error = error

                error_type = get_error_type(
                    error
                )

                logger.error(
                    "Gemini error | model=%s | client=%s | attempt=%s | type=%s | %s",
                    MODEL_NAME,
                    client_index + 1,
                    attempt,
                    error_type,
                    error,
                )

                if error_type == "quota":
                    break

                if error_type in (
                    "auth",
                    "permission",
                ):
                    break

                if error_type == "model":
                    break

                if error_type == "temporary":

                    if attempt < 3:

                        wait_seconds = min(
                            2 ** attempt,
                            8,
                        )

                        wait_seconds += random.uniform(
                            0.2,
                            0.8,
                        )

                        time.sleep(
                            wait_seconds
                        )

                        continue

                    break

                if attempt < 3:
                    time.sleep(2)
                else:
                    break

    raise RuntimeError(
        f"Gemini bilan bog'lanib bo'lmadi: {last_error}"
    )


# ============================================================
# GROQ REQUEST
# ============================================================

def generate_with_groq(
    prompt: str,
):

    if not groq_client:
        raise RuntimeError(
            "GROQ_API_KEY mavjud emas."
        )

    logger.info(
        "Groq request | model=%s",
        GROQ_MODEL,
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    if not response.choices:
        raise RuntimeError(
            "Groq bo'sh javob qaytardi."
        )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "Groq bo'sh javob qaytardi."
        )

    logger.info(
        "Groq success | model=%s",
        GROQ_MODEL,
    )

    return answer.strip()


# ============================================================
# AI ROUTER
# ============================================================

def generate_with_ai_router(
    prompt: str,
):

    # --------------------------------------------------------
    # 1. GEMINI
    # --------------------------------------------------------

    try:

        response = generate_with_gemini(
            prompt
        )

        answer = (
            response.text.strip()
            if response.text
            else ""
        )

        if answer:
            return answer

        raise RuntimeError(
            "Gemini bo'sh javob qaytardi."
        )

    except Exception as gemini_error:

        logger.warning(
            "Gemini ishlamadi. Groq fallback ishga tushadi. %s",
            gemini_error,
        )

    # --------------------------------------------------------
    # 2. GROQ FALLBACK
    # --------------------------------------------------------

    if not groq_client:

        raise RuntimeError(
            "Gemini ishlamadi va GROQ_API_KEY mavjud emas."
        )

    try:

        return generate_with_groq(
            prompt
        )

    except Exception as groq_error:

        logger.exception(
            "Gemini + Groq ikkalasi ham ishlamadi."
        )

        raise RuntimeError(
            "Gemini ham, Groq ham javob bera olmadi."
        ) from groq_error


# ============================================================
# TELEGRAM MESSAGE SPLITTER
# ============================================================

TELEGRAM_MAX_LENGTH = 4000


def split_text(
    text: str,
    max_length: int = TELEGRAM_MAX_LENGTH,
) -> list[str]:

    if not text:
        return [""]

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
) -> None:

    chunks = split_text(text)

    for chunk in chunks:

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
# HEALTH CHECK SERVER
# ============================================================

class HealthCheckHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        self.send_response(
            200
        )

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8",
        )

        self.end_headers()

        self.wfile.write(
            b"Telegram AI is alive."
        )

    def do_HEAD(self):

        self.send_response(
            200
        )

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
# ADMIN CHECK
# ============================================================

def is_admin(
    user_id: int,
) -> bool:

    return user_id == ADMIN_USER_ID


# ============================================================
# /START
# ============================================================

@dp.message(
    CommandStart()
)
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

@dp.message(
    Command("help")
)
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

@dp.message(
    Command("id")
)
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

@dp.message(
    Command("reset")
)
async def reset_handler(
    message: types.Message,
):

    if not message.from_user:
        return

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

@dp.message(
    Command("users")
)
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

    total_users = len(
        users
    )

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

        username = user[
            "username"
        ]

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
# TEXT HANDLER
# ============================================================

@dp.message(
    F.text
)
async def text_handler(
    message: types.Message,
):

    if not message.from_user:
        return

    user_id = message.from_user.id

    register_user(
        message.from_user
    )

    user_text = (
        message.text.strip()
    )

    if not user_text:
        return

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
        user_text[:300],
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

        answer = await loop.run_in_executor(
            None,
            lambda: generate_with_ai_router(
                prompt
            ),
        )

        if not answer:
            answer = (
                "Kechirasiz, javob bo'sh qaytdi."
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
            "Text handler error",
            exc_info=error,
        )

        error_type = get_error_type(
            error
        )

        if error_type == "quota":

            user_error = (
                "⏳ Hozircha AI xizmatlarining "
                "limitiga yetildi.\n\n"
                "Birozdan keyin yana urinib ko'ring."
            )

        elif error_type == "temporary":

            user_error = (
                "⏳ AI serverlari vaqtincha band.\n\n"
                "Bir necha soniyadan keyin yana urinib ko'ring."
            )

        elif error_type in (
            "auth",
            "permission",
        ):

            user_error = (
                "🔐 AI API kaliti yoki ruxsat bilan "
                "bog'liq muammo yuz berdi.\n\n"
                "Administrator API sozlamalarini tekshirishi kerak."
            )

        elif error_type == "model":

            user_error = (
                "⚠️ AI modeli bilan bog'liq muammo yuz berdi.\n\n"
                "Administrator model sozlamasini tekshirishi kerak."
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
# PHOTO HANDLER
# ============================================================

@dp.message(
    F.photo
)
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

        if message.caption:

            prompt = (
                message.caption.strip()
            )

        else:

            prompt = """
Ushbu rasmni diqqat bilan tahlil qiling.

Rasmda nima borligini tushuntiring.

Agar rasmda:
- matn
- savol
- matematika masalasi
- kod
- diagramma
- jadval
- grafik
- hujjat

bo'lsa, uni ham o'qing va tahlil qiling.

Agar foydalanuvchi savol bermagan bo'lsa,
rasm haqida eng muhim va foydali ma'lumotlarni bering.
"""

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
            "Photo handler error",
            exc_info=error,
        )

        error_type = get_error_type(
            error
        )

        if error_type == "quota":

            user_error = (
                "⏳ Hozircha rasm tahlili uchun "
                "Gemini quota limiti tugagan."
            )

        elif error_type == "temporary":

            user_error = (
                "⏳ Gemini serveri vaqtincha band.\n"
                "Birozdan keyin yana urinib ko'ring."
            )

        else:

            user_error = (
                "⚠️ Rasmni tahlil qilishda "
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
# VOICE HANDLER
# ============================================================

@dp.message(
    F.voice
)
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

1. Foydalanuvchining nutqini tushuning.
2. Uning savoli yoki topshirig'ini aniqlang.
3. Agar topshiriq bo'lsa, uni bajaring.
4. Faqat transkripsiya bilan cheklanib qolmang.
5. Javobni foydalanuvchi gapirgan asosiy tilda bering.
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
            "Voice handler error",
            exc_info=error,
        )

        error_type = get_error_type(
            error
        )

        if error_type == "quota":

            user_error = (
                "⏳ Hozircha ovoz tahlili uchun "
                "Gemini quota limiti tugagan."
            )

        elif error_type == "temporary":

            user_error = (
                "⏳ Gemini serveri vaqtincha band.\n"
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
# START BOT
# ============================================================

async def main():

    Thread(
        target=run_health_check_server,
        daemon=True,
    ).start()

    logger.info(
        "========================================"
    )

    logger.info(
        "Telegram AI ishga tushmoqda..."
    )

    logger.info(
        "Gemini model: %s",
        MODEL_NAME,
    )

    logger.info(
        "Thinking level: %s",
        THINKING_LEVEL,
    )

    logger.info(
        "Gemini API clients: %s",
        len(clients),
    )

    logger.info(
        "Groq enabled: %s",
        bool(groq_client),
    )

    logger.info(
        "Groq model: %s",
        GROQ_MODEL,
    )

    logger.info(
        "Admin ID: %s",
        ADMIN_USER_ID,
    )

    logger.info(
        "========================================"
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
