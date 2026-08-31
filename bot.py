import os
import asyncio
import logging
from io import BytesIO
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from PIL import Image
from google import genai
from google.genai import types as genai_types

# Sozlamalar
BOT_TOKEN = "8985071741:AAHZ2palM1JyEdwUSgrvlxYDsctQyw0DLb0"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHANNEL_USERNAME = "@Karnay_uzb"

# AI shaxsiyati va maslahatchilik qobiliyatini maksimal darajaga ko'tarish
SYSTEM_INSTRUCTION = (
    "Siz 'khidirov_ai' — eng kuchli, bilimdon, samimiy va intellektual AI yordamchisiz. "
    "Foydalanuvchilarning har qanday murakkab savollariga chuqur, tahliliy, mantiqiy va "
    "eng to'g'ri amaliy maslahatlarni bera olasiz. Muloqotni ravon o'zbek tilida olib borasiz."
)

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Google GenAI mijozini xavfsiz sozlash
ai_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=genai_types.HttpOptions(
        headers={"x-goog-api-key": GEMINI_API_KEY}
    )
)

async def check_subscription(user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        return member.status in ["creator", "administrator", "member"]
    except Exception:
        return True

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    await message.answer(
        f"Salom, {message.from_user.first_name}! 👋\n\n"
        "Men **khidirov_ai** — sizning eng kuchli AI maslahatchingizman.\n"
        "Menga istalgan savolingizni berishingiz yoki rasm yuborib tahlil qilishingiz mumkin!"
    )

# Matnli xabarlarni qayta ishlash
@dp.message(F.text)
async def ai_text_handler(message: types.Message):
    if not await check_subscription(message.from_user.id):
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📢 Kanalga obuna bo'lish", url=f"https://t.me/{CHANNEL_USERNAME.replace('@', '')}")],
                [InlineKeyboardButton(text="✅ Obunani tekshirish", callback_data="check_sub")]
            ]
        )
        await message.answer("⚠️ Botdan foydalanish uchun kanalimizga obuna bo'ling!", reply_markup=keyboard)
        return

    typing_msg = await message.answer("🤔 Fikr yuritilmoqda...")
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=message.text,
            config={"system_instruction": SYSTEM_INSTRUCTION}
        )
        await typing_msg.edit_text(response.text)
    except Exception as e:
        logging.error(f"Xatolik: {e}")
        await typing_msg.edit_text("Kechirasiz, javob tayyorlashda xatolik yuz berdi.")

# Rasmlarni tahlil qilish (Multimodal AI)
@dp.message(F.photo)
async def ai_photo_handler(message: types.Message):
    if not await check_subscription(message.from_user.id):
        return

    typing_msg = await message.answer("🔍 Rasm tahlil qilinmoqda...")
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        
        image = Image.open(BytesIO(downloaded_file.read()))
        prompt = message.caption if message.caption else "Ushbu rasmni batafsil tahlil qilib bering."

        response = ai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[prompt, image],
            config={"system_instruction": SYSTEM_INSTRUCTION}
        )
        await typing_msg.edit_text(response.text)
    except Exception as e:
        logging.error(f"Rasm xatoligi: {e}")
        await typing_msg.edit_text("Rasmni tahlil qilishda xatolik yuz berdi.")

async def main():
    print("khidirov_ai ishga tushdi...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
