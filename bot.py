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
SYSTEM_INSTRUCTION = (
    "Siz o'ta aqlli, bilimdon va tezkor AI yordamchisiz. "
    "Foydalanuvchining har qanday tildagi (lotin o'zbekcha, kirill o'zbekcha, ruscha va h.k.) "
    "savollariga o'sha tilda mukammal, aniq va chuqur mantiqiy javob bering. "
    "Rasmlarni o'ta aniqlik bilan tahlil qiling va ovozli xabarlardagi fikrni to'liq tushunib javob qaytaring."
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
        "Ассалому алайкум! Мен сизнинг энг кучли АИ ёрдамчингизман.\n"
        "Menga matn (latin/kirill), rasm yoki ovozli xabar yuborishingiz mumkin!"
    )

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
        text_res = response.text if response.text else "Javob olishda muammo bo'ldi."
        await typing_msg.edit_text(text_res, parse_mode=None)
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
        
        image = Image.open(BytesIO(downloaded_file.read()))
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
        text_res = response.text if response.text else "Rasmni tahlil qilib bo'lmadi."
        await typing_msg.edit_text(text_res, parse_mode=None)
    except Exception as e:
        logging.error(f"Rasm xatoligi: {e}")
        await typing_msg.edit_text("Rasmni tahlil qilishda xatolik yuz berdi.")

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
        prompt = "Ushbu ovozli xabarni diqqat bilan eshitib, unda aytilgan savol yoki fikrga batafsil javob ber."
        
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
        text_res = response.text if response.text else "Ovozli xabarni tushunishda muammo bo'ldi."
        await typing_msg.edit_text(text_res, parse_mode=None)
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
