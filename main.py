import random
from time import sleep 

from adb_pywrapper.adb_device import AdbDevice

from bot import Bot

def main():
    bot = Bot(AdbDevice.list_devices()[0])

    if bot.is_offline():
        print("emulator offline")
        exit()

    bot.battle()
    while True:
        sleep(1)
        # bot.screenshot()
        bot.play_card(random.randint(0, 3), random.random(), random.random())

if __name__ == "__main__":
    main()
