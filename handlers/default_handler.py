from misc import dp, vae, bot, nn
from handlers.keyboards import keyboards
from config import Messages
import io
from services import core
from aiogram import types


@dp.message_handler(lambda message: message.text == "About SearchFace bot")
async def bot_info_message(message: types.Message):
    await message.reply(Messages.BOT_INFO, reply_markup=keyboards.start_keyboard)


@dp.message_handler(lambda message: message.text == "About Model")
async def model_info_message(message: types.Message):
    await message.reply(Messages.MODEL_INFO, reply_markup=keyboards.start_keyboard)


@dp.message_handler(content_types=['photo'])
async def get_photo(message: types.Message):
    f = io.BytesIO()
    await message.photo[-1].download(f)
    sample, distance = core.get_similar_photo(f, vae, nn)
    for selected, dist in zip(sample, distance):
        await bot.send_photo(message.from_user.id,
                             core.extract_url(selected),
                             core.extract_caption(selected, dist))




