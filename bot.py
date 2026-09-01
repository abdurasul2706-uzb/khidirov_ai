import asyncio
import os
import logging
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from io import BytesIO

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from google import genai
from google.genai import types as genai_types
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")

# Bir nechta API kalitlarni qo'llab-quvvatlash (Render'da vergul bilan ajratib yoziladi)
RAW_KEYS = os.getenv("GEMINI_API_KEY", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

def get_ai_client():
    """Tasodifiy API kalit tanlab klient yaratadi"""
    if not API_KEYS:
        raise ValueError("GEMINI_API_KEY sozlanmagan!")
    selected_key = random.choice(API_KEYS)
    return genai.Client(api_key=selected_key)

MODEL_NAME = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = (
    "Siz Telegram'dagi eng aqlli, bilimdon, samimiy va tezkor Sun'iy Intellekt yordamchisiz. "
    "Sizning maqsadingiz — har bir foydalanuvchiga (talaba, o'qituvchi, ishchi, dasturchi va boshqalar) xuddi tajribali inson kabi mukammal yordam berish.\n"
    "Qoidalaringiz:\n"
    "1. Foydalanuvchi qaysi tilda yozsa (o'zbekcha lotin/kirill, ruscha, inglizcha va h.k.), aynan o'sha tilda javob bering.\n"
    "2. Javoblaringiz mantiqiy, chuqur, aniq, muloyim va ravon bo'lsin.\n"
    "3. Rasmlar va audio xabarlarni o'ta sinchiklab tahlil qiling va savollarga atroflicha javob bering."
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Rate limit (429) yoki vaqtinchalik xatolarda avtomatik qayta urinish mexanizmi
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_ai_response(contents):
    client = get_ai_client()
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
        )
    )

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

def run_health_check_server():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    await message.answer(
        "Assalomu alaykum! Men sizning universal AI yordamchingizman.\n\n"
        "Menga matn (o'zbek, rus, ingliz tillarida), rasm yoki ovozli xabar yuborishingiz mumkin!"
    )

@dp.message(F.text)
async def text_handler(message: types.Message):
    typing_msg = await message.answer("⚡ Javob tayyorlanmoqda...")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: generate_ai_response(message.text)
        )
        text_res = response.text if response.text else "Javob olishda muammo bo'ldi."
        await typing_msg.edit_text(text_res, parse_mode=None)
    except Exception as e:
        logging.error(f"Matn xatosi: {e}")
        await typing_msg.edit_text("Hozirda AI serverida yuklama yuqori. Bir necha soniyadan so'ng qayta yozib ko'ring.", parse_mode=None)

@dp.message(F.photo)
async def photo_handler(message: types.Message):
    typing_msg = await message.answer("🔍 Rasm tahlil qilinmoqda...")
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        image = Image.open(BytesIO(downloaded_file.read()))
        prompt = message.caption if message.caption else "Ushbu rasmni batafsil tahlil qilib ber."
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: generate_ai_response([prompt, image])
        )
        text_res = response.text if response.text else "Rasmni tahlil qilib bo'lmadi."
        await typing_msg.edit_text(text_res, parse_mode=None)
    except Exception as e:
        logging.error(f"Rasm xatosi: {e}")
        await typing_msg.edit_text("Rasmni tahlil qilishda xatolik yuz berdi.", parse_mode=None)

@dp.message(F.voice)
async def voice_handler(message: types.Message):
    typing_msg = await message.answer("🎙 Ovozli xabar eshitilmoqda...")
    try:
        voice = message.voice
        file_info = await bot.get_file(voice.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        audio_part = genai_types.Part.from_bytes(
            data=downloaded_file.read(),
            mime_type="audio/ogg"
        )
        prompt = "Ushbu ovozli xabarni eshitib, undagi savolga batafsil javob ber."
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: generate_ai_response([prompt, audio_part])
        )
        text_res = response.text if response.text else "Ovozli xabarni tushunishda muammo bo'ldi."
        await typing_msg.edit_text(text_res, parse_mode=None)
    except Exception as e:
        logging.error(f"Ovoz xatosi: {e}")
        await typing_msg.edit_text("Ovozli xabarni tahlil qilishda xatolik yuz berdi.", parse_mode=None)

async def main():
    Thread(target=run_health_check_server, daemon=True).start()
    logging.info("AI Bot ishga tushdi...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
