from time import sleep
from adb_pywrapper.adb_device import AdbDevice

from ClashAI.bot import Bot
from ClashAI.vision import get_full_game_state
from ClashAI.game_state import GameState


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

    print("\nStarting vision data collection (3s intervals)...")

    try:
        while True:
            # 1. Analyze screenshot and get the game state
            game_state: GameState = get_full_game_state(bot)

            # 2. Print results using the __str__ method
            print("--------------------")
            print(game_state)
            
            sleep(3)

    except KeyboardInterrupt:
        print("\nVision test stopped by user.")


if __name__ == "__main__":
    main()
