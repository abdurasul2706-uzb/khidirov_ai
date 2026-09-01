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

MODEL_NAME = "gemini-3.6-flash"
SYSTEM_INSTRUCTION = "Siz o'ta aqlli va bilimdon yordamchisiz. Foydalanuvchi savollariga o'zbek tilida aniq, ravon va londa javob bering."

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
    await message.answer("Salom! Men sizning aqlli AI yordamchingizman. Menga matn, rasm yoki ovozli xabar yuborishingiz mumkin!")

@dp.message(F.text)
async def text_handler(message: types.Message):
    typing_msg = await message.answer("⚡ Javob tayyorlanmoqda...")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ai_client.models.generate_content(
                model=MODEL_NAME,
                contents=message.text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )
        )
        await typing_msg.edit_text(response.text if response.text else "Javob olishda muammo bo'ldi.")
    except Exception as e:
        logging.error(f"Matn xatoligi: {e}")
        await typing_msg.edit_text("Kechirasiz, javob tayyorlashda xatolik yuz berdi.")

@dp.message(F.photo)
async def photo_handler(message: types.Message):
    typing_msg = await message.answer("🔍 Rasm tahlil qilinmoqda...")
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        image_bytes = downloaded_file.read()
        image = Image.open(BytesIO(image_bytes))
        prompt = message.caption if message.caption else "Ushbu rasmni batafsil tahlil qilib ber."
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ai_client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, image],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )
        )
        await typing_msg.edit_text(response.text if response.text else "Rasmni tahlil qilib bo'lmadi.")
    except Exception as e:
        logging.error(f"Rasm xatoligi: {e}")
        await typing_msg.edit_text("Rasmni tahlil qilishda xatolik yuz berdi.")

@dp.message(F.voice)
async def voice_handler(message: types.Message):
    typing_msg = await message.answer("🎙 Ovozli xabar eshitilmoqda va tahlil qilinmoqda...")
    try:
        voice = message.voice
        file_info = await bot.get_file(voice.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        audio_bytes = downloaded_file.read()
        
        # Audio ma'lumotni Gemini API ga part sifatida uzatish
        audio_part = genai_types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/ogg"
        )
        
        prompt = "Ushbu ovozli xabarda nima deyilganini tushunib, unga batafsil javob ber."
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ai_client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, audio_part],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )
        )
        await typing_msg.edit_text(response.text if response.text else "Ovozli xabarni tushunishda muammo bo'ldi.")
    except Exception as e:
        logging.error(f"Ovoz xatoligi: {e}")
        await typing_msg.edit_text("Ovozli xabarni tahlil qilishda xatolik yuz berdi.")

async def main():
    Thread(target=run_health_check_server, daemon=True).start()
    logging.info("khidirov_ai ishga tushdi...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
