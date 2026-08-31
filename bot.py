import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from google import genai
from google.genai import types as genai_types

BOT_TOKEN = "8985071741:AAHZ2palM1JyEdwUSgrvlxYDsctQyw0DLb0"
# Google AI Studio bergan AQ.Ab8RN6... kalitingizni shu yerga qo'ying:
 import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHANNEL_USERNAME = "@Karnay_uzb"

SYSTEM_INSTRUCTION = (
    "Siz o'zbek tilida muloqot qiluvchi intellektual va samimiy AI yordamchisiz. "
    "Foydalanuvchilarning barcha savollariga aniq, ravon va foydali javob berasiz."
)

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Google GenAI mijozini HTTP header orqali avtorizatsiya qilish sozlamasi
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
        "Men o'zbek tilidagi shaxsiy AI yordamchingizman. "
        "Menga istalgan savolingizni yuborishingiz mumkin!"
    )

@dp.message()
async def ai_message_handler(message: types.Message):
    user_id = message.from_user.id
    is_subscribed = await check_subscription(user_id)
    if not is_subscribed:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📢 Kanalga obuna bo'lish", url=f"https://t.me/{CHANNEL_USERNAME.replace('@', '')}")],
                [InlineKeyboardButton(text="✅ Obunani tekshirish", callback_data="check_sub")]
            ]
        )
        await message.answer(
            "⚠️ **Botdan foydalanish uchun avval kanalimizga obuna bo'ling!**",
            reply_markup=keyboard,
            parse_mode="Markdown"
        )
        return

    typing_msg = await message.answer("🤔 Yozilmoqda...")
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=message.text,
            config={"system_instruction": SYSTEM_INSTRUCTION}
        )
        await typing_msg.edit_text(response.text)
    except Exception as e:
        logging.error(f"Xatolik: {e}")
        await typing_msg.edit_text("Kechirasiz, xatolik yuz berdi.")

async def main():
    print("Bot ishga tushdi...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
