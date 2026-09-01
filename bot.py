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

# Gemini 1.5 Flash - Rasmiy, multimodal va eng tezkor model
model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
    system_instruction=(
        "Siz Telegram'dagi eng aqlli va bilimdon sun'iy intellekt yordamchisiz. "
        "Foydalanuvchi qaysi tilda yozsa (o'zbekcha lotin/kirill, ruscha, inglizcha), "
        "aynan o'sha tilda an'anaviy va mukammal javob bering."
    )
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

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
    await message.answer("Assalomu alaykum! Men sizning AI yordamchingizman. Savol, rasm yoki ovozli xabaringizni yuboring!")

@dp.message(F.text)
async def text_handler(message: types.Message):
    typing_msg = await message.answer("⚡ Javob tayyorlanmoqda...")
    try:
        # Asinxron so'rov bajarish
        res = await asyncio.to_thread(model.generate_content, message.text)
        await typing_msg.edit_text(res.text if res.text else "Javob bo'sh qaytdi.", parse_mode=None)
    except Exception as e:
        logging.error(f"Matn xatosi: {e}")
        await typing_msg.edit_text(f"API Xatoligi: {str(e)[:200]}", parse_mode=None)

@dp.message(F.photo)
async def photo_handler(message: types.Message):
    typing_msg = await message.answer("🔍 Rasm tahlil qilinmoqda...")
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        image = Image.open(BytesIO(downloaded_file.read()))
        prompt = message.caption if message.caption else "Ushbu rasmni batafsil tahlil qilib ber."
        
        res = await asyncio.to_thread(model.generate_content, [prompt, image])
        await typing_msg.edit_text(res.text if res.text else "Rasm bo'yicha javob olinmadi.", parse_mode=None)
    except Exception as e:
        logging.error(f"Rasm xatosi: {e}")
        await typing_msg.edit_text(f"Rasm xatoligi: {str(e)[:200]}", parse_mode=None)

@dp.message(F.voice)
async def voice_handler(message: types.Message):
    typing_msg = await message.answer("🎙 Ovozli xabar eshitilmoqda...")
    try:
        voice = message.voice
        file_info = await bot.get_file(voice.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        audio_data = {
            "mime_type": "audio/ogg",
            "data": downloaded_file.read()
        }
        prompt = "Ushbu ovozli xabardagi gaplarni tinglab, to'liq javob ber."
        
        res = await asyncio.to_thread(model.generate_content, [prompt, audio_data])
        await typing_msg.edit_text(res.text if res.text else "Ovoz bo'yicha javob olinmadi.", parse_mode=None)
    except Exception as e:
        logging.error(f"Ovoz xatosi: {e}")
        await typing_msg.edit_text(f"Ovoz xatoligi: {str(e)[:200]}", parse_mode=None)

async def main():
    Thread(target=run_health_check_server, daemon=True).start()
    logging.info("Bot ishga tushdi...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
