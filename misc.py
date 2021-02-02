from aiogram import Bot, Dispatcher, types
from config import Config
from model import loader, VAE
from sklearn.neighbors import NearestNeighbors

bot = Bot(token=Config.TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)

vae, nn = loader.load_models()
