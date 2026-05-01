from time import sleep
from typing import Optional

import dotenv
from adb_pywrapper.adb_device import AdbDevice

from .agent import ActorCritic
from .bot import Bot
from .game_state import GameState
from .logger import *
from .vision import CARD_CONTINUOUS_DIM, FIXED_INPUT_DIM, num_card_types

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
    info("=" * 5 + " ARW " + "=" * 5)

    # --- Initialize ADB, Bots, and Agent ---
    devices = AdbDevice.list_devices()
    if not devices:
        error("ERROR: No ADB devices found.")
        exit()

    bots = []
    for device_id in devices:
        debug(f"Connecting to ADB device: {device_id}...")
        bot = Bot(device_id)

        # --- Verify Minicap Installation ---
        if not bot.minicap.is_installed():
            error(
                f"ERROR: Minicap is not installed or functional on device {device_id}."
            )
            bot.check_minicap()
            continue

        bots.append(bot)

    if not bots:
        error("ERROR: No functional bots initialized.")
        exit(1)

    agent = ActorCritic(
        fixed_input_dim=FIXED_INPUT_DIM,
        num_card_types=num_card_types,
        card_continuous_feature_dim=CARD_CONTINUOUS_DIM,
        num_discrete_actions=NUM_DISCRETE_ACTIONS,
        continuous_action_dim=CONTINUOUS_ACTION_DIM,
        card_embedding_size=CARD_EMBEDDING_SIZE,
    )
    agent.load_model()
    # Ensure model is in training mode
    agent.train()
    info(f"Reinforcement learning agent initialized. Using {len(bots)} devices.")

    debug("\nStarting bot...")

    # Track previous state for each bot
    previous_states: dict[str, Optional[GameState]] = {bot.device: None for bot in bots}

    try:
        while True:
            for bot in bots:
                # The agent's step function now handles the full game loop logic for one device
                previous_states[bot.device] = agent.step(
                    bot, previous_states[bot.device]
                )

            # Add a small delay between rounds of steps
            sleep(0.1)

    except KeyboardInterrupt:
        info("\nBot stopped by user.")
    finally:
        # Save the model on exit
        agent.save_model()
