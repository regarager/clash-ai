import json  # Added to read cards.json
import os
from time import sleep
from typing import Any, cast

import dotenv
import torch

from adb_pywrapper.adb_device import AdbDevice

from agent import ActorCritic
from bot import Bot
from roboflow_service import get_roboflow_predictions  # Added

# Load environment variables from .env file at the very beginning
dotenv.load_dotenv()

# --- Model and State Configuration ---
FIXED_INPUT_DIM = 3
# The following are derived from the downloaded model's data.yaml
num_card_types: int | None = None
class_names: list[str] | None = None
CARD_CONTINUOUS_DIM = 4  # [center_x, center_y, width, height] from YOLO output
NUM_DISCRETE_ACTIONS = 5
CONTINUOUS_ACTION_DIM = 2
CARD_EMBEDDING_SIZE = 16

# Load CLASS_NAMES from cards.json
try:
    with open("cards.json", "r") as f:
        cards_data = json.load(f)
        class_names = [card["name"] for card in cards_data["items"]]
        num_card_types = len(class_names)
    print(f"Loaded {num_card_types} class names from cards.json.")
except Exception as e:
    print(f"Error loading class names from cards.json: {e}")
    exit()


def get_state(
    bot: Bot,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """
    Gets the current game state by taking a screenshot and running the Roboflow
    object detection model via API.
    """
    print("Capturing screenshot and running Roboflow detection...")
    screenshot_path = bot.screenshot()
    if not os.path.exists(screenshot_path):
        print(f"Error: Screenshot file not found at {screenshot_path}")
        return (
            torch.randn(FIXED_INPUT_DIM),
            torch.empty(0, dtype=torch.long),
            torch.empty(0),
            [],
        )

    # 1. Run Roboflow object detection model via API
    try:
        roboflow_results = get_roboflow_predictions(screenshot_path)

        detections: list[dict[str, Any]] = roboflow_results.get(
            "predictions", []
        )  # Assuming predictions key in roboflow_results
    except Exception as e:
        print(f"Error during Roboflow API prediction: {e}")
        return (
            torch.randn(FIXED_INPUT_DIM),
            torch.empty(0, dtype=torch.long),
            torch.empty(0),
            [],
        )
    finally:
        os.remove(screenshot_path)

    print(f"Detected {len(detections)} objects.")
    print(f"Roboflow Detections: {detections}")

    # 2. Simulate fixed inputs (placeholder)
    fixed_inputs = torch.randn(FIXED_INPUT_DIM)

    # 3. Preprocess detections into tensors
    if len(detections) == 0:
        return fixed_inputs, torch.empty(0, dtype=torch.long), torch.empty(0), []

    card_ids_list: list[int] = []
    card_continuous_features_list: list[list[float]] = []

    # Get screen size for normalization
    screen_width, screen_height = bot.get_screen_size()

    for detection in detections:
        class_name = detection["class"]
        try:
            card_id = cast(list[str], class_names).index(class_name)
            card_ids_list.append(card_id)

            x_center = detection["x"] / screen_width
            y_center = detection["y"] / screen_height
            width = detection["width"] / screen_width
            height = detection["height"] / screen_height

            card_continuous_features_list.append([x_center, y_center, width, height])
        except ValueError:
            print(
                f"Warning: Detected class '{class_name}' not found in CLASS_NAMES. Skipping."
            )
            continue

    if not card_ids_list:  # If no valid cards were detected
        return fixed_inputs, torch.empty(0, dtype=torch.long), torch.empty(0), []

    card_ids = torch.tensor(card_ids_list, dtype=torch.long)
    card_continuous_features = torch.tensor(
        card_continuous_features_list, dtype=torch.float32
    )

    return fixed_inputs, card_ids, card_continuous_features, detections


def calculate_reward(
    bot: Bot,
    current_detections: list[dict[str, Any]],
    previous_detections: list[dict[str, Any]],
) -> float:
    """
    Calculates the reward based on the change in the number of detected towers.

    Args:
        bot (Bot): The bot instance, for context.
        current_detections (list): The list of detections from the current state.
        previous_detections (list): The list of detections from the previous state.

    Returns:
        float: The calculated reward.
    """
    reward = 0.0

    # Define class names for towers (these are assumptions and may need to be adjusted)
    # Based on the Roboflow model's actual output.
    # We will assume a naming convention like 'ally_king_tower', 'ally_princess_tower',
    # 'enemy_king_tower', 'enemy_princess_tower'.
    ally_tower_classes = {"ally_king_tower", "ally_princess_tower"}
    enemy_tower_classes = {"enemy_king_tower", "enemy_princess_tower"}

    # --- Count towers in current and previous states ---

    current_ally_towers = sum(
        1 for d in current_detections if d.get("class") in ally_tower_classes
    )
    previous_ally_towers = sum(
        1 for d in previous_detections if d.get("class") in ally_tower_classes
    )

    current_enemy_towers = sum(
        1 for d in current_detections if d.get("class") in enemy_tower_classes
    )
    previous_enemy_towers = sum(
        1 for d in previous_detections if d.get("class") in enemy_tower_classes
    )

    # --- Reward for destroying towers ---
    if current_enemy_towers < previous_enemy_towers:
        print("Reward: +5 (Enemy tower destroyed)")
        reward += 5.0

    # --- Penalty for losing towers ---
    if current_ally_towers < previous_ally_towers:
        print("Reward: -5 (Ally tower lost)")
        reward -= 5.0

    # --- Small continuous rewards/penalties ---
    # Small penalty for each enemy tower still standing
    reward -= 0.1 * current_enemy_towers

    # Small reward for each ally tower still standing
    reward += 0.1 * current_ally_towers

    if reward != 0.0:
        print(f"Calculated reward: {reward}")
    else:
        print("Calculated reward: 0.0 (No change in towers)")

    return reward


def main() -> None:
    """
    Main loop for the Clash Royale AI bot using the Roboflow API.
    """
    print("--- Clash Royale AI Bot (Roboflow API Inference) ---")

    # --- Initialize ADB, Bot, and Agent ---
    try:
        bot = Bot(AdbDevice.list_devices()[0])
    except IndexError:
        print("ERROR: No ADB devices found.")
        exit()
    print(f"Connected to ADB device: {bot.device}.")

    agent = ActorCritic(
        fixed_input_dim=FIXED_INPUT_DIM,
        num_card_types=cast(int, num_card_types),
        card_continuous_feature_dim=CARD_CONTINUOUS_DIM,
        num_discrete_actions=NUM_DISCRETE_ACTIONS,
        continuous_action_dim=CONTINUOUS_ACTION_DIM,
        card_embedding_size=CARD_EMBEDDING_SIZE,
    )
    agent.load_model()  # Load existing model if available
    agent.train()  # Set the agent to training mode
    print("Reinforcement learning agent initialized.")

    print("\nStarting bot...")

    previous_detections: list[dict[str, Any]] = []  # Initialize previous detections

    try:
        while True:
            # 1. Get state using the Roboflow API
            (
                fixed_state,
                card_ids,
                card_features,
                current_detections,
            ) = get_state(bot)

            # 2. Ask agent for action and store log_probs and state_value
            d_action, c_action = agent.select_action(
                fixed_state, card_ids, card_features
            )

            # 3. Take action
            if d_action < 4:
                print(f"Agent chose to play card {d_action} at position {c_action}")
                scaled_x = (c_action[0] + 1) / 2
                scaled_y = (c_action[1] + 1) / 2
                bot.play_card(d_action, scaled_x, scaled_y)
            else:
                print("Agent chose to do nothing (action 4).")

            sleep(2)  # Wait for the game to update

            # 4. Calculate reward
            reward: float = calculate_reward(
                bot, current_detections, previous_detections
            )
            agent.rewards.append(reward)

            # 5. Update the agent
            agent.update()

            previous_detections = (
                current_detections  # Update previous detections for next iteration
            )

    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        # Save the model on exit
        agent.save_model()


if __name__ == "__main__":
    main()