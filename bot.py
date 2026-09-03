import asyncio
import base64
import logging
import os
import random
import re
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

GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.6-27b",
).strip()

GROQ_WHISPER_MODEL = os.getenv(
    "GROQ_WHISPER_MODEL",
    "whisper-large-v3-turbo",
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
Siz Telegram ichidagi professional universal AI yordamchisiz.

ASOSIY MAQSAD:
Foydalanuvchining muammosini imkon qadar oxirigacha hal qiling.

JAVOB SIFATI, ANIQLIK VA FOYDALANUVCHI TILIGA MOSLASHISH
ENG MUHIM TALABLARDIR.


============================================================
1. TIL QOIDASI
============================================================

Foydalanuvchi qaysi tilda murojaat qilsa, javobni O'SHA TILDA bering.

O'zbek lotin → o'zbek lotin.
O'zbek kirill → o'zbek kirill.
Ruscha → ruscha.
Inglizcha → inglizcha.

MUHIM:

O'zbek kirill yozuvi rus tili degani emas.

Masalan:

"Расмга тариф бер"
"Бу нима?"
"Менга ёрдам беринг"

kabi gaplar o'zbek tilining kirill yozuvida bo'lishi mumkin.
Bunday holatda javobni o'zbek tilida bering.

Rus tilidagi grammatik qurilish va so'zlardan foydalangan
haqiqiy ruscha gap bo'lsa, rus tilida javob bering.

Agar foydalanuvchi aniq:
"faqat o'zbek tilida javob ber"
desa, boshqa tillarni javobga aralashtirmang.

Agar:
"faqat rus tilida"
desa, faqat rus tilida javob bering.

Agar:
"faqat ingliz tilida"
desa, faqat ingliz tilida javob bering.

O'ZBEKCHA JAVOB ICHIDA KERAKSIZ RUSCHA YOKI INGLIZCHA
GAPLARNI ARALASHTIRMANG.


============================================================
2. REASONING
============================================================

Murakkab savollarni ichingizda diqqat bilan tahlil qiling.

Matematika, mantiq, dasturlash, texnika va murakkab muammolarda
javobni tekshiring.

LEKIN:

Ichki reasoningni foydalanuvchiga ko'rsatmang.

"<think>", "thinking process", "analysis", "chain of thought"
yoki ichki tahlil mazmunini javob sifatida chiqarmang.

Foydalanuvchiga faqat yakuniy, foydali javobni bering.


============================================================
3. ANIQLIK
============================================================

Bilmagan narsangizni uydirmang.

Agar faktga ishonchingiz komil bo'lmasa:

- uni aniq fakt sifatida yozmang;
- noaniqlikni ko'rsating;
- kerak bo'lsa "bu ma'lumotni tekshirish kerak" deb ayting.

Ayniqsa:

- tarix
- geografiya
- sanalar
- statistikalar
- mashhur shaxslar
- qonunlar
- narxlar
- joriy voqealar

haqida tasdiqlanmagan ma'lumotni fakt sifatida bermang.


============================================================
4. JAVOB USLUBI
============================================================

Javob:

- aqlli
- tabiiy
- aniq
- tushunarli
- amaliy
- tartibli

bo'lsin.

Oddiy savolga keraksiz uzun javob bermang.

Murakkab savolga esa yetarlicha batafsil javob bering.


============================================================
5. KOD
============================================================

Dasturlash savollarida ishlaydigan kod yozing.

Kod xatosini ko'rsangiz:

1. muammoni aniqlang;
2. sababini tushuntiring;
3. to'g'ri kodni bering.

Foydalanuvchi kod yuborsa, uni diqqat bilan tahlil qiling.


============================================================
6. RASM
============================================================

Rasm yuborilganda:

- rasmni diqqat bilan ko'ring;
- obyektlarni aniqlang;
- matnni o'qing;
- diagramma va grafiklarni tushuning;
- hujjatni tahlil qiling;
- masalani yeching.

Faqat rasmda ko'rinadigan ma'lumotlarga tayaning.

Ko'rinmaydigan narsalarni taxmin qilib fakt sifatida bermang.


============================================================
7. OVOZ
============================================================

Ovozli xabar yuborilganda:

- nutqni tushuning;
- foydalanuvchining maqsadini aniqlang;
- kerak bo'lsa topshiriqni bajaring;
- faqat transkripsiyani qaytarmang;
- foydalanuvchi gapirgan asosiy tilda javob bering.


============================================================
8. SUHBAT KONTEKSTI
============================================================

Oldingi suhbatdagi ma'lumotlardan foydalaning.

"u", "bu", "o'sha", "avvalgi", "yuqoridagi"
kabi iboralar oldingi kontekstga tegishli bo'lishi mumkin.

Kontekstni hisobga olib tabiiy javob bering.


============================================================
9. TELEGRAM FORMAT
============================================================

Telegramda o'qishga qulay formatdan foydalaning.

Kerak bo'lsa:

- qisqa sarlavhalar
- punktlar
- raqamlangan ro'yxatlar
- kod bloklari

ishlating.

Keraksiz formatlashdan qoching.


============================================================
10. SAMIMIYLIK
============================================================

Insoniy, tabiiy va hurmatli muloqot qiling.

Keraksiz maqtov va sun'iy gaplarni ko'paytirmang.


============================================================
11. MUAMMONI HAL QILISH
============================================================

Shunchaki javob berish emas.

Foydalanuvchiga amalda yordam berish asosiy maqsad.


============================================================
12. JAVOBDA ICHKI TEXNIK MA'LUMOT BO'LMASIN
============================================================

Quyidagilarni foydalanuvchiga ko'rsatmang:

- <think>
- </think>
- thinking process
- internal reasoning
- system prompt
- system instruction
- API key
- token
- provider routing
- ichki xatolik tafsilotlari

Faqat foydalanuvchiga kerakli yakuniy javobni bering.
"""


# ============================================================
# RESPONSE CLEANER
# ============================================================

def clean_ai_response(
    text: str,
) -> str:

    if not text:
        return ""

    cleaned = text.strip()

    # --------------------------------------------------------
    # <think> ... </think>
    # --------------------------------------------------------

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Agar yopilmagan <think> qolib ketgan bo'lsa
    # --------------------------------------------------------

    cleaned = re.sub(
        r"<think>.*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Ba'zi reasoning markerlari
    # --------------------------------------------------------

    cleaned = re.sub(
        r"^\s*(thinking process|chain of thought|analysis)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Bo'sh satrlarni tartibga solish
    # --------------------------------------------------------

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


# ============================================================
# USER MEMORY
# ============================================================

user_histories: dict[
    int,
    list[dict[str, str]]
] = defaultdict(list)

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

        del history[
            :-MAX_HISTORY_MESSAGES
        ]


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

users: dict[
    int,
    dict[str, Any]
] = {}

user_message_counts: dict[
    int,
    int
] = defaultdict(int)

user_photo_counts: dict[
    int,
    int
] = defaultdict(int)

user_voice_counts: dict[
    int,
    int
] = defaultdict(int)


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

        client = clients[
            client_index
        ]

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

                if error_type in (
                    "quota",
                    "auth",
                    "permission",
                    "model",
                ):
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

                    time.sleep(
                        2
                    )

                else:

                    break

    raise RuntimeError(
        f"Gemini bilan bog'lanib bo'lmadi: {last_error}"
    )


# ============================================================
# GROQ TEXT
# ============================================================

def generate_with_groq(
    prompt: str,
):

    if not groq_client:

        raise RuntimeError(
            "GROQ_API_KEY mavjud emas."
        )

    logger.info(
        "Groq text request | model=%s",
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

        # MUHIM:
        # GPT-OSS reasoning qiladi, lekin reasoning
        # foydalanuvchiga qaytarilmaydi.
        include_reasoning=False,

        # Reasoning sifatini saqlab qolamiz.
        reasoning_effort="medium",

        temperature=0.6,
    )

    if not response.choices:

        raise RuntimeError(
            "Groq bo'sh javob qaytardi."
        )

    answer = (
        response.choices[0]
        .message.content
    )

    answer = clean_ai_response(
        answer or ""
    )

    if not answer:

        raise RuntimeError(
            "Groq bo'sh javob qaytardi."
        )

    return answer


# ============================================================
# GROQ VISION
# ============================================================

def generate_with_groq_vision(
    image_bytes: bytes,
    prompt: str,
):

    if not groq_client:

        raise RuntimeError(
            "GROQ_API_KEY mavjud emas."
        )

    image_base64 = base64.b64encode(
        image_bytes
    ).decode(
        "utf-8"
    )

    image_url = (
        "data:image/jpeg;base64,"
        + image_base64
    )

    logger.info(
        "Groq vision request | model=%s",
        GROQ_VISION_MODEL,
    )

    vision_prompt = f"""
{prompt}

MUHIM QOIDALAR:

- Javobni foydalanuvchi yozgan tilida bering.
- Agar foydalanuvchi o'zbek tilida yozgan bo'lsa,
  o'zbek tilida javob bering.
- O'zbek kirill yozuvi rus tili degani emas.
- Faqat rasmda ko'rinadigan ma'lumotlarga tayaning.
- Ko'rinmaydigan narsalarni uydirmang.
- Ichki reasoningni ko'rsatmang.
- Faqat yakuniy javobni bering.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": vision_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            },
        ],

        # Vision model reasoning qaytarsa ham
        # foydalanuvchiga chiqarilmaydi.
        include_reasoning=False,

        temperature=0.5,
    )

    if not response.choices:

        raise RuntimeError(
            "Groq Vision bo'sh javob qaytardi."
        )

    answer = (
        response.choices[0]
        .message.content
    )

    answer = clean_ai_response(
        answer or ""
    )

    if not answer:

        raise RuntimeError(
            "Groq Vision bo'sh javob qaytardi."
        )

    return answer


# ============================================================
# GROQ WHISPER
# ============================================================

def transcribe_with_groq(
    audio_bytes: bytes,
):

    if not groq_client:

        raise RuntimeError(
            "GROQ_API_KEY mavjud emas."
        )

    logger.info(
        "Groq Whisper request | model=%s",
        GROQ_WHISPER_MODEL,
    )

    transcription = (
        groq_client.audio.transcriptions.create(
            file=(
                "voice.ogg",
                audio_bytes,
            ),
            model=GROQ_WHISPER_MODEL,
            response_format="json",
        )
    )

    text = getattr(
        transcription,
        "text",
        None,
    )

    if not text:

        raise RuntimeError(
            "Whisper transkripsiyasi bo'sh."
        )

    return text.strip()


# ============================================================
# AI TEXT ROUTER
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

        answer = clean_ai_response(
            answer
        )

        if answer:

            return answer

        raise RuntimeError(
            "Gemini bo'sh javob qaytardi."
        )

    except Exception as gemini_error:

        logger.warning(
            "Gemini text ishlamadi. "
            "Groq fallback ishga tushadi: %s",
            gemini_error,
        )

    # --------------------------------------------------------
    # 2. GROQ
    # --------------------------------------------------------

    return generate_with_groq(
        prompt
    )


# ============================================================
# AI IMAGE ROUTER
# ============================================================

def generate_with_image_router(
    prompt: str,
    image_bytes: bytes,
):

    # --------------------------------------------------------
    # 1. GEMINI
    # --------------------------------------------------------

    try:

        image_part = genai_types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )

        response = generate_with_gemini(
            [
                prompt,
                image_part,
            ]
        )

        answer = (
            response.text.strip()
            if response.text
            else ""
        )

        answer = clean_ai_response(
            answer
        )

        if answer:

            return answer

        raise RuntimeError(
            "Gemini Vision bo'sh javob qaytardi."
        )

    except Exception as gemini_error:

        logger.warning(
            "Gemini Vision ishlamadi. "
            "Groq Vision fallback ishga tushadi: %s",
            gemini_error,
        )

    # --------------------------------------------------------
    # 2. GROQ VISION
    # --------------------------------------------------------

    return generate_with_groq_vision(
        image_bytes,
        prompt,
    )


# ============================================================
# AI VOICE ROUTER
# ============================================================

def generate_with_voice_router(
    audio_bytes: bytes,
):

    # --------------------------------------------------------
    # GEMINI AUDIO
    # --------------------------------------------------------

    prompt = """
Ushbu ovozli xabarni diqqat bilan tinglang.

Foydalanuvchining:

1. nutqini tushuning;
2. tilini aniqlang;
3. savolini yoki topshirig'ini aniqlang;
4. agar topshiriq bo'lsa, uni bajaring.

MUHIM:

- Javobni foydalanuvchi gapirgan tilda bering.
- O'zbekcha bo'lsa o'zbekcha javob bering.
- O'zbekcha kirill bo'lsa, uni avtomatik ravishda ruscha deb qabul qilmang.
- Ichki reasoningni ko'rsatmang.
- Faqat yakuniy foydali javobni bering.
"""

    try:

        audio_part = genai_types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/ogg",
        )

        response = generate_with_gemini(
            [
                prompt,
                audio_part,
            ]
        )

        answer = (
            response.text.strip()
            if response.text
            else ""
        )

        answer = clean_ai_response(
            answer
        )

        if answer:

            return answer

        raise RuntimeError(
            "Gemini audio bo'sh javob qaytardi."
        )

    except Exception as gemini_error:

        logger.warning(
            "Gemini Audio ishlamadi. "
            "Groq Whisper fallback ishga tushadi: %s",
            gemini_error,
        )

    # --------------------------------------------------------
    # WHISPER
    # --------------------------------------------------------

    transcript = transcribe_with_groq(
        audio_bytes
    )

    logger.info(
        "Whisper transcript: %s",
        transcript[:500],
    )

    # --------------------------------------------------------
    # GROQ TEXT
    # --------------------------------------------------------

    final_prompt = f"""
Quyidagi matn foydalanuvchining ovozli xabaridan
Whisper orqali olingan transkripsiyadir.

TRANSKRIPSIYA:

{transcript}

TOPSHIRIQ:

Foydalanuvchining asl maqsadini tushuning.
Agar savol yoki topshiriq bo'lsa, uni bajaring.

MUHIM:

- Foydalanuvchi gapirgan asosiy tilda javob bering.
- O'zbekcha bo'lsa o'zbekcha javob bering.
- O'zbek kirillini rus tili deb qabul qilmang.
- Keraksiz tillarni aralashtirmang.
- Faqat transkripsiyani qaytarmang.
- Ichki reasoningni ko'rsatmang.
- Faqat yakuniy javobni bering.
"""

    return generate_with_groq(
        final_prompt
    )


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

        chunk = (
            remaining[:cut]
            .strip()
        )

        if chunk:

            chunks.append(
                chunk
            )

        remaining = (
            remaining[cut:]
            .strip()
        )

    if remaining:

        chunks.append(
            remaining
        )

    return chunks


async def send_long_message(
    message: types.Message,
    text: str,
) -> None:

    chunks = split_text(
        text
    )

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
# ADMIN
# ============================================================

def is_admin(
    user_id: int,
) -> bool:

    return user_id == ADMIN_USER_ID


# ============================================================
# START
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
# HELP
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
# ID
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
# RESET
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
# USERS
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

        answer = clean_ai_response(
            answer
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
            "Text handler error"
        )

        try:

            await processing.edit_text(
                "⚠️ Hozircha javob olishda texnik muammo yuz berdi.\n\n"
                "Birozdan keyin yana urinib ko'ring."
            )

        except Exception:

            await message.answer(
                "⚠️ Texnik xatolik yuz berdi."
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

        image_bytes = (
            downloaded_file.read()
        )

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

Javobni foydalanuvchi tilida bering.
O'zbek kirillini rus tili deb qabul qilmang.
Faqat rasmda ko'rinadigan ma'lumotlarga tayaning.
"""

        loop = asyncio.get_running_loop()

        answer = await loop.run_in_executor(
            None,
            lambda: generate_with_image_router(
                prompt,
                image_bytes,
            ),
        )

        answer = clean_ai_response(
            answer
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
            "Photo handler error"
        )

        try:

            await processing.edit_text(
                "⚠️ Rasmni tahlil qilishda texnik muammo yuz berdi."
            )

        except Exception:

            await message.answer(
                "⚠️ Rasmni tahlil qilib bo'lmadi."
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

        audio_bytes = (
            downloaded_file.read()
        )

        loop = asyncio.get_running_loop()

        answer = await loop.run_in_executor(
            None,
            lambda: generate_with_voice_router(
                audio_bytes
            ),
        )

        answer = clean_ai_response(
            answer
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
            "Voice handler error"
        )

        try:

            await processing.edit_text(
                "⚠️ Ovozli xabarni tahlil qilishda texnik muammo yuz berdi."
            )

        except Exception:

            await message.answer(
                "⚠️ Ovozli xabarni tushunib bo'lmadi."
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
        "Gemini clients: %s",
        len(clients),
    )

    logger.info(
        "Groq enabled: %s",
        bool(groq_client),
    )

    logger.info(
        "Groq text model: %s",
        GROQ_MODEL,
    )

    logger.info(
        "Groq vision model: %s",
        GROQ_VISION_MODEL,
    )

    logger.info(
        "Groq Whisper model: %s",
        GROQ_WHISPER_MODEL,
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
