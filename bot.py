if __name__ == "__main__":
    from misc import dp
    from aiogram.utils import executor
    import handlers

    executor.start_polling(dp, skip_updates=True)
