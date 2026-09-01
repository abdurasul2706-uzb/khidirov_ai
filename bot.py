import asyncio
import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from io import BytesIO

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from google import genai
from PIL import Image

# Logging sozlamalari
logging.basicConfig(level=logging.INFO)

# Token va API kalitlarini olish
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Render port timeout xatoligini oldini olish uchun oddiy HTTP server
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_health_check_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# /start komandasi
@dp.message(CommandStart())
async def start_handler(message: types.Message):
    await message.answer("Salom! Men khidirov_ai botiman. Savolingizni yuboring yoki rasm jo'nating!")

# Matnli xabarlarga javob berish (Gemini 1.5 Flash)
@dp.message(F.text)
async def text_handler(message: types.Message):
    typing_msg = await message.answer("🤔 Fikr yuritilmoqda...")
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=message.text
        )
        await typing_msg.edit_text(response.text)
    except Exception as e:
        logging.error(f"Xatolik: {e}")
        await typing_msg.edit_text("Kechirasiz, javob tayyorlashda xatolik yuz berdi.")

# Rasmlarni tahlil qilish (Multimodal AI)
@dp.message(F.photo)
async def photo_handler(message: types.Message):
    typing_msg = await message.answer("🔍 Rasm tahlil qilinmoqda...")
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        image = Image.open(BytesIO(downloaded_file.read()))
        prompt = message.caption if message.caption else "Ushbu rasmni batafsil tasvirlab ber."
        
        response = ai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )
        await typing_msg.edit_text(response.text)
    except Exception as e:
        logging.error(f"Rasm xatoligi: {e}")
        await typing_msg.edit_text("Rasmni tahlil qilishda xatolik yuz berdi.")

async def main():
    # Render portini uxlab qolmasligi uchun fonda ishga tushirish
    Thread(target=run_health_check_server, daemon=True).start()
    logging.info("khidirov_ai ishga tushdi...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
