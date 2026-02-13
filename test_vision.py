import os
from time import sleep

from adb_pywrapper.adb_device import AdbDevice

from bot import Bot
from vision import get_full_game_state
from game_state import GameState  # Import GameState


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
            # 1. Analyze screenshot and get the game state
            game_state: GameState = get_full_game_state(bot)

            # 2. Print results using the __str__ method
            print("--------------------")
            print(game_state)

            # The screenshot is cleaned up inside get_full_game_state
            # sleep(1)  # Wait 1 second before next capture

    except KeyboardInterrupt:
        print("\nVision test stopped by user.")


if __name__ == "__main__":
    main()
