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

# Muhim:
# whisper-large-v3 multilingual ovozlar uchun kuchliroq.
GROQ_WHISPER_MODEL = os.getenv(
    "GROQ_WHISPER_MODEL",
    "whisper-large-v3",
).strip()

# Asosiy foydalanuvchi tili Uzbek.
# "auto" yoki bo'sh qoldirilsa avtomatik aniqlashga o'tadi.
GROQ_WHISPER_LANGUAGE = os.getenv(
    "GROQ_WHISPER_LANGUAGE",
    "uz",
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
    ADMIN_USER_ID = int(
        ADMIN_USER_ID_RAW
    )
except ValueError:
    raise RuntimeError(
        "ADMIN_USER_ID noto'g'ri."
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
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
Siz Telegram ichidagi universal, kuchli va ishonchli AI yordamchisiz.

ASOSIY MAQSAD:
Foydalanuvchining muammosini imkon qadar oxirigacha hal qiling.

Javob sifati, aniqlik, foydalilik va halollik birinchi o'rinda.


============================================================
1. TILNI QAT'IY SAQLASH
============================================================

Javob tilini ENG SO'NGGI foydalanuvchi xabaridan aniqlang.

Qoidalar:

- O'zbek lotin → O'zbek lotin.
- O'zbek kirill → O'zbek kirill.
- Rus tili → Rus tili.
- Ingliz tili → Ingliz tili.
- Boshqa til → imkon qadar o'sha til.

O'zbek lotin tilida javob berayotganingizda:

- qozoqcha so'zlarni aralashtirmang
- ozarbayjoncha so'zlarni aralashtirmang
- turkcha so'zlarni aralashtirmang
- ruscha so'zlarni keraksiz ishlatmang

Agar foydalanuvchi O'zbek tilida gapirsa,
javobni tabiiy O'zbek tilida bering.

Oldingi suhbat boshqa tilda bo'lgan bo'lsa,
ENG SO'NGGI xabar tilini ustun qo'ying.


============================================================
2. REASONING MAXFIYLIGI
============================================================

Murakkab savollarni ichingizda chuqur tahlil qiling.

Matematika:
- hisobni tekshiring
- natijani qayta tekshiring

Dasturlash:
- kodni tahlil qiling
- xatolarni tekshiring
- ishlaydigan yechim bering

Mantiq:
- shartlarni tekshiring
- taxminni fakt sifatida bermang

MUHIM:

Quyidagilarni foydalanuvchiga ko'rsatmang:

<think>
<analysis>
<reasoning>
chain of thought
ichki fikrlash
ichki reasoning
modelning yashirin tahlili

Faqat yakuniy foydali javobni ko'rsating.


============================================================
3. FAKTUAL ANIQLIK
============================================================

Bilmagan narsangizni UYDIRMANG.

Ayniqsa:

- tarix
- sana
- yil
- statistika
- narx
- reyting
- ism
- manzil
- qonun
- yangilik
- geografiya

bo'yicha ishonchingiz bo'lmasa,
aniq fakt sifatida yozmang.

Agar aniq bilmasangiz:

"Bu ma'lumotni aniq tasdiqlay olmayman."

yoki

"Bu bo'yicha ishonchim yetarli emas."

deb ayting.

Uydirma javobdan ko'ra halol javob yaxshi.


============================================================
4. RASM TAHLILI — JUDA MUHIM
============================================================

Rasmni tahlil qilayotganda faqat rasmda
HAQIQATAN KO'RINAYOTGAN ma'lumotlarga tayaning.

Quyidagilarni o'ylab topmang:

- aniq yil
- aniq sana
- odamning ismi
- odamning kimligi
- aniq manzil
- aniq joy
- voqeaning tarixi
- kamera modeli
- rasmning olingan sanasi
- odamning aniq yoshi
- rasm muallifi

Agar bunday ma'lumot rasmning o'zida aniq ko'rinmasa,
"aniq ko'rinmaydi" deb ayting.

Masalan:

Noto'g'ri:
"Bu rasm 2007-yilda olingan."

Agar rasmda buni ko'rsatuvchi dalil bo'lmasa.

To'g'ri:
"Rasmning o'zidan aniq yilni aniqlab bo'lmaydi."

Rasmda odamlar bo'lsa,
ularning shaxsini aniqlashga urinmang.

Ko'rinayotgan narsani tavsiflang.


============================================================
5. RASMDA MATN
============================================================

Agar rasmda matn bo'lsa:

- matnni diqqat bilan o'qing
- ko'rinmaydigan qismini o'ylab topmang
- noaniq joyni noaniq deb belgilang

Agar:

- kod
- formula
- matematika
- jadval
- diagramma
- grafik
- hujjat

bo'lsa, uni tahlil qiling.


============================================================
6. OVOZ
============================================================

Ovozli xabar uchun:

- nutqni tushuning
- asosiy tilni aniqlang
- transkripsiya xatosi bo'lishi mumkinligini hisobga oling
- foydalanuvchining asl maqsadini tushuning
- faqat transkripsiyani qaytarmang
- topshiriqni bajaring

Agar transkripsiya shubhali bo'lsa,
kontekst asosida eng ehtimoliy ma'noni aniqlang.

Lekin ma'noni asossiz o'zgartirmang.


============================================================
7. DASTURLASH
============================================================

Kod yuborilsa:

1. Kodni tahlil qiling.
2. Muammoni toping.
3. Sababini tushuntiring.
4. To'g'ri kodni bering.
5. Kerak bo'lsa ishga tushirishni ko'rsating.

Kod bloklaridan foydalaning.


============================================================
8. SUHBAT KONTEKSTI
============================================================

Oldingi suhbatdan foydalaning.

Masalan:

"u"
"bu"
"o'sha"
"avvalgi"
"yuqoridagi"

kabi iboralarni kontekst orqali tushuning.

Ammo yangi xabar eski ma'lumotga zid bo'lsa,
yangi xabarni ustun qo'ying.


============================================================
9. JAVOB USLUBI
============================================================

Javob:

- tabiiy
- aniq
- aqlli
- foydali
- amaliy
- tushunarli

bo'lsin.

Oddiy savolga qisqa javob bering.

Murakkab savolga yetarlicha batafsil javob bering.

Keraksiz maqtov va ortiqcha gaplardan qoching.


============================================================
10. TELEGRAM FORMAT
============================================================

Telegram uchun qulay format:

- punktlar
- raqamlangan ro'yxatlar
- qisqa sarlavhalar
- kod bloklari

Keraksiz formatlashdan qoching.


============================================================
11. ENG MUHIM QOIDA
============================================================

Shunchaki javob yozmang.

Foydalanuvchining ASL MAQSADINI tushuning
va unga amalda foydali yechim bering.
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

    # Think blocks
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned = re.sub(
        r"<reasoning>.*?</reasoning>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Unclosed blocks
    for tag in (
        "<think>",
        "<analysis>",
        "<reasoning>",
    ):
        position = cleaned.lower().find(tag)

        if position != -1:
            cleaned = cleaned[:position]

    # Common prefixes
    cleaned = re.sub(
        r"^\s*analysis\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^\s*reasoning\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Ba'zi modellar "final answer" kabi belgilar chiqarishi mumkin.
    cleaned = re.sub(
        r"^\s*final answer\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned.strip()


# ============================================================
# LANGUAGE HINT
# ============================================================

def detect_language_hint(
    text: str,
) -> str:

    lower = text.lower()

    # Uzbek Cyrillic
    if any(
        char in lower
        for char in "ўқғҳ"
    ):
        return "UZBEK_CYRILLIC"

    words = set(
        re.findall(
            r"[a-zA-ZА-Яа-яЎўҚқҒғҲҳЁёЪъ]+",
            lower,
        )
    )

    uzbek_words = {
        "va",
        "uchun",
        "bilan",
        "men",
        "siz",
        "bu",
        "shu",
        "qanday",
        "nima",
        "haqida",
        "kerak",
        "qilib",
        "qil",
        "menga",
        "senga",
        "qayerda",
        "qachon",
        "nega",
        "bo'ladi",
        "boladi",
        "menga",
        "biz",
        "bizga",
        "sizga",
        "ayting",
        "bering",
        "mumkin",
        "emas",
        "bor",
        "yo'q",
        "yoq",
    }

    russian_words = {
        "что",
        "как",
        "почему",
        "зачем",
        "можно",
        "нужно",
        "это",
        "для",
        "меня",
        "тебя",
        "пожалуйста",
        "расскажи",
        "объясни",
        "помоги",
    }

    english_words = {
        "the",
        "what",
        "how",
        "why",
        "please",
        "about",
        "can",
        "could",
        "would",
        "help",
        "with",
        "give",
        "tell",
        "explain",
    }

    if len(words & uzbek_words) >= 1:
        return "UZBEK_LATIN"

    if len(words & russian_words) >= 1:
        return "RUSSIAN"

    if len(words & english_words) >= 1:
        return "ENGLISH"

    return "AUTO"


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
        del history[:-MAX_HISTORY_MESSAGES]


def build_prompt(
    user_id: int,
    current_message: str,
) -> str:

    history = user_histories.get(
        user_id,
        [],
    )

    language_hint = detect_language_hint(
        current_message
    )

    if not history:

        return f"""
OUTPUT LANGUAGE LOCK: {language_hint}

Javobni ENG SO'NGGI foydalanuvchi xabarining
tilida bering.

Foydalanuvchi xabari:

{current_message}
"""

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
OUTPUT LANGUAGE LOCK: {language_hint}

MUHIM:
Javob tilini faqat ENG SO'NGGI foydalanuvchi
xabaridan aniqlang.

Oldingi suhbat boshqa tilda bo'lgan bo'lsa ham,
yangi foydalanuvchi xabarining tilida javob bering.

OLDINGI SUHBAT:

{history_text}

---

YANGI FOYDALANUVCHI XABARI:

{current_message}

---

Oldingi suhbatni hisobga olib,
yangi xabarga aniq, tabiiy va foydali javob bering.
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
                        "AI PROVIDER=GEMINI | success | model=%s | client=%s",
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

                    time.sleep(2)

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

    try:

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
            extra_body={
                "include_reasoning": False,
            },
        )

    except TypeError as error:

        logger.warning(
            "Groq extra_body qabul qilinmadi. "
            "Oddiy request ishlatiladi: %s",
            error,
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

    answer = (
        response.choices[0]
        .message.content
    )

    if not answer:

        raise RuntimeError(
            "Groq bo'sh javob qaytardi."
        )

    answer = clean_ai_response(
        answer
    )

    if not answer:

        raise RuntimeError(
            "Groq javobidan faqat reasoning chiqdi."
        )

    logger.info(
        "AI PROVIDER=GROQ | text success | model=%s",
        GROQ_MODEL,
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
    ).decode("utf-8")

    image_url = (
        "data:image/jpeg;base64,"
        + image_base64
    )

    strict_vision_prompt = f"""
RASM TAHLILI QOIDALARI:

Faqat rasmda HAQIQATAN KO'RINAYOTGAN narsalar
asosida javob bering.

Hech qachon dalilsiz quyidagilarni o'ylab topmang:

- aniq sana
- aniq yil
- ism
- shaxsning kimligi
- aniq manzil
- aniq joy
- voqea tarixi
- kamera modeli
- rasm olingan vaqt

Agar ma'lumot rasmda ko'rinmasa:

"Bu rasmning o'zidan aniqlanmaydi."

deb ayting.

Odamlarning shaxsini aniqlamang.

Agar taxmin qilayotgan bo'lsangiz,
uni FAKT sifatida emas, taxmin sifatida belgilang.

Foydalanuvchining topshirig'iga birinchi navbatda javob bering.

FOYDALANUVCHI TOPSHIRIG'I:

{prompt}
"""

    logger.info(
        "Groq vision request | model=%s",
        GROQ_VISION_MODEL,
    )

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
                        "text": strict_vision_prompt,
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
    )

    if not response.choices:

        raise RuntimeError(
            "Groq Vision bo'sh javob qaytardi."
        )

    answer = (
        response.choices[0]
        .message.content
    )

    if not answer:

        raise RuntimeError(
            "Groq Vision bo'sh javob qaytardi."
        )

    answer = clean_ai_response(
        answer
    )

    if not answer:

        raise RuntimeError(
            "Groq Vision javobi bo'sh."
        )

    logger.info(
        "AI PROVIDER=GROQ | vision success | model=%s",
        GROQ_VISION_MODEL,
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
        "Groq Whisper request | model=%s | language=%s",
        GROQ_WHISPER_MODEL,
        GROQ_WHISPER_LANGUAGE or "auto",
    )

    request_kwargs = {
        "file": (
            "voice.ogg",
            audio_bytes,
        ),
        "model": GROQ_WHISPER_MODEL,
        "response_format": "json",
    }

    # "auto" yoki bo'sh bo'lsa language yuborilmaydi.
    if GROQ_WHISPER_LANGUAGE not in (
        "",
        "auto",
        "none",
    ):
        request_kwargs["language"] = (
            GROQ_WHISPER_LANGUAGE
        )

    transcription = (
        groq_client.audio.transcriptions.create(
            **request_kwargs
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

    text = text.strip()

    logger.info(
        "AI PROVIDER=GROQ | Whisper success | model=%s | text=%s",
        GROQ_WHISPER_MODEL,
        text[:300],
    )

    return text


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
            "Gemini text ishlamadi. Groq fallback: %s",
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
    # 1. GEMINI VISION
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
            "Gemini Vision ishlamadi. Groq Vision fallback: %s",
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
    # 1. GEMINI AUDIO
    # --------------------------------------------------------

    prompt = """
Ushbu ovozli xabarni diqqat bilan tushuning.

1. Foydalanuvchining nutqini tushuning.
2. Asosiy tilni aniqlang.
3. Savol yoki topshiriqni aniqlang.
4. Agar topshiriq bo'lsa, uni bajaring.
5. Faqat transkripsiyani qaytarmang.
6. Javobni foydalanuvchining asosiy tilida bering.
7. Ichki reasoningni ko'rsatmang.
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
            "Gemini Audio ishlamadi. Groq Whisper fallback: %s",
            gemini_error,
        )

    # --------------------------------------------------------
    # 2. WHISPER
    # --------------------------------------------------------

    transcript = transcribe_with_groq(
        audio_bytes
    )

    logger.info(
        "Whisper transcript: %s",
        transcript[:500],
    )

    # --------------------------------------------------------
    # 3. GROQ TEXT
    # --------------------------------------------------------

    final_prompt = f"""
VOICE INPUT MODE

Quyidagi matn foydalanuvchining ovozli xabaridan
Whisper orqali olingan.

Transkripsiyada xatolar bo'lishi mumkin.

TRANSKRIPSIYA:

{transcript}

VAZIFA:

Foydalanuvchining asl maqsadini tushuning.

Agar savol bo'lsa:
→ savolga javob bering.

Agar topshiriq bo'lsa:
→ topshiriqni bajaring.

Agar foydalanuvchi Uzbek tilida gapirgan bo'lsa:
→ O'zbek tilida javob bering.

Agar transkripsiyada Ozarbayjoncha yoki boshqa
turkiy tilga o'xshash noto'g'ri so'zlar paydo bo'lsa,
kontekstni tekshiring va foydalanuvchining ehtimoliy
asl O'zbekcha ma'nosini tiklashga harakat qiling.

Lekin foydalanuvchi ma'nosini o'zboshimchalik bilan
o'zgartirmang.

Faqat transkripsiyani qaytarmang.

Ichki reasoningni chiqarmang.
<think>, <analysis>, <reasoning> chiqarmang.
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
            chunks.append(chunk)

        remaining = (
            remaining[cut:]
            .strip()
        )

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

Faqat rasmda ko'rinadigan ma'lumotlarga tayaning.

Agar ma'lumot aniq ko'rinmasa,
uni o'ylab topmang.

Agar foydalanuvchi savol bermagan bo'lsa,
rasm haqida eng muhim va foydali ma'lumotlarni bering.

Javobni foydalanuvchining asosiy tilida bering.
"""

        loop = asyncio.get_running_loop()

        answer = await loop.run_in_executor(
            None,
            lambda: generate_with_image_router(
                prompt,
                image_bytes,
            ),
        )

        add_to_history(
            user_id,
            "user",
            message.caption or "[Rasm yuborildi]",
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

        add_to_history(
            user_id,
            "user",
            "[Ovozli xabar]",
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
        "Groq Whisper language: %s",
        GROQ_WHISPER_LANGUAGE or "auto",
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
