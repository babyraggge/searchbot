from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

button_about_bot = KeyboardButton("About SearchFace bot")
button_about_model = KeyboardButton("About Model")

start_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
start_keyboard.add(button_about_bot, button_about_model)