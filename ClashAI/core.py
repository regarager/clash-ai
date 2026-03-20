from time import sleep
from typing import Optional

import dotenv
from adb_pywrapper.adb_device import AdbDevice

from .agent import ActorCritic
from .bot import Bot
from .vision import FIXED_INPUT_DIM, num_card_types, CARD_CONTINUOUS_DIM
from .game_state import GameState

# Load environment variables from .env file at the very beginning
dotenv.load_dotenv()

# --- Model and State Configuration ---
NUM_DISCRETE_ACTIONS = 5
CONTINUOUS_ACTION_DIM = 2
CARD_EMBEDDING_SIZE = 16


def run() -> None:
    """
    Main loop for the Clash Royale AI bot.
    """
    print("--- Clash Royale AI Bot ---")

    # --- Initialize ADB, Bot, and Agent ---
    try:
        # Assuming the first device is the one we want to use
        bot = Bot(AdbDevice.list_devices()[0])
    except IndexError:
        print("ERROR: No ADB devices found.")
        exit()

    print(f"Connected to ADB device: {bot.device}.")

    # --- Verify Minicap Installation ---
    if not bot.minicap.is_installed():
        print("ERROR: Minicap is not installed or functional on the device.")
        bot.check_minicap()  # Run diagnostic check for user help
        exit(1)

    agent = ActorCritic(
        bot=bot,
        fixed_input_dim=FIXED_INPUT_DIM,
        num_card_types=num_card_types,
        card_continuous_feature_dim=CARD_CONTINUOUS_DIM,
        num_discrete_actions=NUM_DISCRETE_ACTIONS,
        continuous_action_dim=CONTINUOUS_ACTION_DIM,
        card_embedding_size=CARD_EMBEDDING_SIZE,
    )
    agent.load_model()
    agent.train()
    print("Reinforcement learning agent initialized.")

    print("\nStarting bot...")

    previous_state: Optional[GameState] = None

    try:
        while True:
            # The agent's step function now handles the full game loop logic
            previous_state = agent.step(previous_state)
            # Add a delay to control the loop speed
            sleep(1)

    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        # Save the model on exit
        agent.save_model()
