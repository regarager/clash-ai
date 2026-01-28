import os
from time import sleep

from adb_pywrapper.adb_device import AdbDevice

from bot import Bot
from vision import get_elixir, get_tower_healths, is_elixir_bar_visible


def main() -> None:
    """
    Main loop for testing the vision system.
    """
    print("--- Clash Royale Vision Test ---")

    # --- Initialize ADB and Bot ---
    try:
        bot = Bot(AdbDevice.list_devices()[0])
    except IndexError:
        print("ERROR: No ADB devices found.")
        exit()
    print(f"Connected to ADB device: {bot.device}.")

    print("\nStarting vision test...")

    try:
        while True:
            # 1. Get screenshot
            screenshot_path = bot.screenshot()
            if not os.path.exists(screenshot_path):
                print(f"Error: Screenshot file not found at {screenshot_path}")
                continue

            # 2. Analyze screenshot
            elixir = get_elixir(screenshot_path)
            tower_healths = get_tower_healths(screenshot_path)
            elixir_bar_visible = is_elixir_bar_visible(screenshot_path)

            # 3. Print results
            print(f"Elixir: {elixir}")
            print(f"Tower Healths: {tower_healths}")
            print(f"Elixir Bar Visible: {elixir_bar_visible}")
            
            # 4. Clean up screenshot
            os.remove(screenshot_path)

            sleep(1)  # Wait 1 second before next capture

    except KeyboardInterrupt:
        print("\nVision test stopped by user.")


if __name__ == "__main__":
    main()

