import asyncio
import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from io import BytesIO

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
import google.generativeai as genai
from PIL import Image

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# API ni sozlash
genai.configure(api_key=GEMINI_API_KEY)

# Dunyodagi eng kuchli multimodal model sozlamalari
SYSTEM_INSTRUCTION = (
    "Siz Telegram'dagi eng kuchli, donishmand, samimiy va tezkor Sun'iy Intelekt yordamchisiz. "
    "Sizning maqsadingiz — har bir foydalanuvchiga (talaba, o'qituvchi, ishchi, dasturchi) xuddi tajribali inson kabi mukammal yordam berish.\n"
    "Qoidalaringiz:\n"
    "1. Muloqot tili: Foydalanuvchi qaysi tilda yozsa (lotin o'zbekcha, kirill o'zbekcha, ruscha, inglizcha va h.k.), aynan o'sha tilda javob bering.\n"
    "2. Uslub: Mantiqiy, chuqur, aniq, muloyim va foydali bo'lsin. Murakkab narsalarni ham oddiy va tushunarli tilda tushuntiring.\n"
    "3. Rasmlar va audio: Rasmlarni o'ta sinchiklab tahlil qiling. Ovozli xabar kelsa, undagi har bir so'z va ma'noni to'g'ri anglab javob bering.\n"
    "4. Formatlash: Telegram belgilari bilan xatolik kelib chiqmasligi uchun javobingizni chiroyli va tartibli matn shaklida berishga harakat qiling."
)

model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
    system_instruction=SYSTEM_INSTRUCTION
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Render bepul serveri uxlab qolmasligi uchun Health Check
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
    welcome_text = (
        "Assalomu alaykum! Men sizning universal AI yordamchingizman.\n\n"
        "Menga o'zingizni qiziqtirgan har qanday savolni berishingiz mumkin:\n"
        " Matnli savollar (Lotin/Kirill, Rus, Ingliz tillarida)\n"
        " Rasmlar (tahlil qilish va yechim topshirish uchun)\n"
        " Ovozli xabarlar\n\n"
        "Sizga yordam berishdan xursandman!"
    )
    await message.answer(welcome_text)

@dp.message(F.text)
async def text_handler(message: types.Message):
    typing_msg = await message.answer("⚡ Javob o'ylanmoqda...")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(message.text)
        )
        text_res = response.text if response.text else "Kechirasiz, fikrimni shakllantirishda muammo bo'ldi."
        await typing_msg.edit_text(text_res, parse_mode=None)
    except Exception as e:
        logging.error(f"Matn xatosi: {e}")
        await typing_msg.edit_text("Hozirda so'rovlar juda ko'payib ketdi. Iltimos, bir ozdan so'ng qayta urinib ko'ring.")

@dp.message(F.photo)
async def photo_handler(message: types.Message):
    typing_msg = await message.answer("🔍 Rasm tahlil qilinmoqda...")
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        image = Image.open(BytesIO(downloaded_file.read()))
        prompt = message.caption if message.caption else "Ushbu rasmni chuqur tahlil qilib, unga batafsil javob ber."
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content([prompt, image])
        )
        text_res = response.text if response.text else "Rasmni to'liq tahlil qila olmadim."
        await typing_msg.edit_text(text_res, parse_mode=None)
    except Exception as e:
        logging.error(f"Rasm xatosi: {e}")
        await typing_msg.edit_text("Rasmni qayta ishlashda xatolik bo'ldi. Qayta yuborib ko'ring.")

@dp.message(F.voice)
async def voice_handler(message: types.Message):
    typing_msg = await message.answer("🎙 Ovozingiz eshitilmoqda...")
    try:
        voice = message.voice
        file_info = await bot.get_file(voice.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        audio_data = {
            "mime_type": "audio/ogg",
            "data": downloaded_file.read()
        }
        prompt = "Ushbu ovozli xabarda nima deyilganini diqqat bilan tinglab, unga o'ta aniq va atroflicha javob ber."
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content([prompt, audio_data])
        )
        text_res = response.text if response.text else "Ovozli xabarni tushunishda muammo bo'ldi."
        await typing_msg.edit_text(text_res, parse_mode=None)
    except Exception as e:
        logging.error(f"Ovoz xatosi: {e}")
        await typing_msg.edit_text("Ovozli xabarni tahlil qilishda texnik xatolik yuz berdi.")

async def main():
    Thread(target=run_health_check_server, daemon=True).start()
    logging.info("AI Bot muvaffaqiyatli ishga tushdi...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
