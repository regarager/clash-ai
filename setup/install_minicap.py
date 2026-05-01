import sys
import os

from ClashAI.logger import *

# Add root project dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ClashAI.bot import Bot
from adb_pywrapper.adb_device import AdbDevice


def main():
    try:
        devices = AdbDevice.list_devices()
        if not devices:
            error("No ADB devices found.")
            return

        device_id = devices[0]
        bot = Bot(device_id)

        info(f"--- Minicap Setup (Device: {device_id}) ---")
        abi = bot.minicap.get_abi()
        sdk = bot.minicap.get_sdk()
        debug(f"Detected ABI: {abi}")
        debug(f"Detected SDK: {sdk}")

        info(
            "\nTo install minicap, you need two files from the DeviceFarmer/minicap repository:"
        )
        info(f"1. bin/{abi}/minicap")
        info(f"2. shared/android-{sdk}/{abi}/minicap.so")

        info(f"\nExample push command:")
        info(f"adb -s {device_id} push bin/{abi}/minicap /data/local/tmp/")
        info(
            f"adb -s {device_id} push shared/android-{sdk}/{abi}/minicap.so /data/local/tmp/"
        )
        info(f"adb -s {device_id} shell chmod 777 /data/local/tmp/minicap")

        debug("\nRunning diagnostic check...")
        bot.check_minicap()

    except Exception as e:
        error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
