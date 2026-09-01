import asyncio
import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from io import BytesIO

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from google import genai
from google.genai import types as genai_types
from PIL import Image

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
ai_client = genai.Client(api_key=GEMINI_API_KEY)

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_health_check_server():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    await message.answer("Salom! Men khidirov_ai botiman. Savolingizni yuboring!")

@dp.message(F.text)
async def text_handler(message: types.Message):
    typing_msg = await message.answer("⚡ Javob tayyorlanmoqda...")
    try:
        # Asinxron ravishda AI javobini olish (to'xtab qolmasligi uchun)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ai_client.models.generate_content(
                model="models/gemini-3.6-flash",
                contents=message.text,
                config=genai_types.GenerateContentConfig(
                    system_instruction="Javobni o'zbek tilida, aniq va tushunarli uslubda ber."
                )
            )
        )
        await typing_msg.edit_text(response.text)
    except Exception as e:
        logging.error(f"Text error: {e}")
        await typing_msg.edit_text("Kechirasiz, javob tayyorlashda xatolik yuz berdi.")

@dp.message(F.photo)
async def photo_handler(message: types.Message):
    typing_msg = await message.answer("🔍 Rasm tahlil qilinmoqda...")
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        image = Image.open(BytesIO(downloaded_file.read()))
        prompt = message.caption if message.caption else "Ushbu rasmni tasvirlab ber."
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ai_client.models.generate_content(
                model="models/gemini-3.6-flash",
                contents=[prompt, image]
            )
        )
        await typing_msg.edit_text(response.text)
    except Exception as e:
        logging.error(f"Photo error: {e}")
        await typing_msg.edit_text("Rasmni tahlil qilishda xatolik yuz berdi.")

async def main():
    Thread(target=run_health_check_server, daemon=True).start()
    logging.info("khidirov_ai ishga tushdi...")
    # Har safar eski navbatlarni tozalab ishga tushadi
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
