import sys
import os

# Add root project dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ClashAI.bot import Bot
from adb_pywrapper.adb_device import AdbDevice


def main():
    try:
        devices = AdbDevice.list_devices()
        if not devices:
            print("No ADB devices found.")
            return
        
        device_id = devices[0]
        bot = Bot(device_id)
        
        print(f"--- Minicap Setup (Device: {device_id}) ---")
        abi = bot.minicap.get_abi()
        sdk = bot.minicap.get_sdk()
        print(f"Detected ABI: {abi}")
        print(f"Detected SDK: {sdk}")
        
        print("\nTo install minicap, you need two files from the DeviceFarmer/minicap repository:")
        print(f"1. bin/{abi}/minicap")
        print(f"2. shared/android-{sdk}/{abi}/minicap.so")
        
        print(f"\nExample push command:")
        print(f"adb -s {device_id} push bin/{abi}/minicap /data/local/tmp/")
        print(f"adb -s {device_id} push shared/android-{sdk}/{abi}/minicap.so /data/local/tmp/")
        print(f"adb -s {device_id} shell chmod 777 /data/local/tmp/minicap")
        
        print("\nRunning diagnostic check...")
        bot.check_minicap()
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
