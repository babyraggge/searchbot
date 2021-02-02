from misc import dp
from handlers.keyboards import keyboards
from config import Messages

from aiogram import types


@dp.message_handler(commands=["start"])
async def start_message(message: types.Message):
    user_first_name = message.from_user.first_name
    await message.answer(f"Welcome {user_first_name}! " + Messages.START_MESSAGE,
                        reply_markup=keyboards.start_keyboard)


@dp.message_handler(commands=["help"])
async def help_message(message: types.Message):
    await message.answer(Messages.HELP_MESSAGE,
                        reply_markup=keyboards.start_keyboard)
