import os
import time  # For timestamp in debug image filename
from typing import Any, Optional

import cv2
import numpy as np
import torch

from bot import Bot
from cards import CARDS, TOWERS
from game_state import GameState
from local_yolo_service import get_yolo_predictions
from positions import ELIXIR_BAR_BBOX, TOWER_BBOXES, BBox

# --- Model and State Configuration Constants (moved from main.py) ---
FIXED_INPUT_DIM = 9  # Elixir, 6 Tower Healths, Ally Unit Count, Enemy Unit Count
# The following are derived from the downloaded model's data.yaml
CLASS_NAMES = sorted(
    list(
        set(
            [card.split("-")[-1] for card in CARDS]
            + [tower.split("-")[-1] for tower in TOWERS]
        )
    )
)
num_card_types: int = len(CLASS_NAMES)
class_names: list[str] = CLASS_NAMES
CARD_CONTINUOUS_DIM = 4  # [center_x, center_y, width, height] from YOLO output


def _debug_save_health_bboxes(
    image_path: str, tower_bboxes: dict[str, BBox], suffix: str = ""
):
    """
    Loads an image, draws bounding boxes for tower health bars, and saves the debug image.
    """
    # TODO: remove print
    print("saving debug ss")
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"DEBUG: Could not load image for drawing bboxes: {image_path}")
            return

        for name, bbox in tower_bboxes.items():
            x1, y1, x2, y2 = bbox.to_xyxy()
            color = (
                (0, 255, 0) if "ally" in name else (0, 0, 255)
            )  # Green for ally, Red for enemy
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Draw rectangle

            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (255, 255, 255)  # White text
            cv2.putText(
                img,
                name,
                (x1, y1 - 5),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

        debug_filename = f"debug_health_bboxes_{int(time.time())}{suffix}.png"
        debug_path = os.path.join(os.path.dirname(image_path), debug_filename)
        cv2.imwrite(debug_path, img)
        print(f"DEBUG: Saved health bar bboxes visualization to {debug_path}")
    except Exception as e:
        print(f"DEBUG ERROR: Failed to save debug health bboxes image: {e}")


def _get_bar_fill_percentage(
    bar_image: np.ndarray, color_ranges: list[tuple[np.ndarray, np.ndarray]]
) -> Optional[float]:
    """
    Calculates the percentage of a bar that is filled with a specific color.
    Returns a float (0.0-1.0) if filled, or None if the bar has no pixels.
    """
    hsv_image = cv2.cvtColor(bar_image, cv2.COLOR_BGR2HSV)

    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv_image, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    filled_pixels = cv2.countNonZero(combined_mask)
    total_pixels = bar_image.shape[0] * bar_image.shape[1]

    if total_pixels == 0:
        return None

    return filled_pixels / total_pixels

def get_elixir(image_path: str) -> int:
    """
    Calculates the current elixir count from a screenshot of the game.
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0

    # Bounding box for the elixir bar
    x, y, w, h = ELIXIR_BAR_BBOX.to_xywh()
    elixir_bar_image = image[y : y + h, x : x + w]

    # Purple color range in HSV
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])

    fill_ratio = _get_bar_fill_percentage(
        elixir_bar_image, [(lower_purple, upper_purple)]
    )

    if fill_ratio is None:
        return 0

    # Convert the fill ratio to an elixir value (0-10)
    elixir = int(round(fill_ratio * 10))

    return elixir


def debug_color_range(image_path: str, bbox: BBox, tower_name: str):
    """Debug function to find HSV values for empty portion of health bar."""
    image = cv2.imread(image_path)
    x, y, w, h = bbox.to_xywh()
    health_bar = image[y : y + h, x : x + w]
    hsv = cv2.cvtColor(health_bar, cv2.COLOR_BGR2HSV)

    # Sample from the rightmost 5 pixels
    sample_region = hsv[:, -3:, :]
    sample_pixels = sample_region.reshape(-1, 3)

    print(f"\n--- {tower_name} ---")
    print(f"Sample region shape: {sample_region.shape}")
    print("First 10 pixel values (HSV):")
    for i, pixel in enumerate(sample_pixels[:9]):
        print(f"  {i}: H={pixel[0]:3.0f}, S={pixel[1]:3.0f}, V={pixel[2]:3.0f}")

    mean_hsv = np.mean(sample_pixels, axis=0)
    std_hsv = np.std(sample_pixels, axis=0)
    min_hsv = np.min(sample_pixels, axis=0)
    max_hsv = np.max(sample_pixels, axis=0)

    print("\nStats:")
    print(f"  Mean: H={mean_hsv[0]:.1f}, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")
    print(f"  Std:  H={std_hsv[0]:.1f}, S={std_hsv[1]:.1f}, V={std_hsv[2]:.1f}")
    print(f"  Min:  H={min_hsv[0]:.0f}, S={min_hsv[1]:.0f}, V={min_hsv[2]:.0f}")
    print(f"  Max:  H={max_hsv[0]:.0f}, S={max_hsv[1]:.0f}, V={max_hsv[2]:.0f}")

    # Check BGR values too
    bgr_sample = health_bar[:, -3:, :].reshape(-1, 3)
    bgr_mean = np.mean(bgr_sample, axis=0)
    print(f"\nBGR Mean: B={bgr_mean[0]:.1f}, G={bgr_mean[1]:.1f}, R={bgr_mean[2]:.1f}")


def get_tower_healths(image_path: str) -> dict[str, float]:
    """
    Calculates the health of each tower from a screenshot of the game.
    Returns a dictionary with tower names as keys and health (0.0-1.0) as values.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {}

    _debug_save_health_bboxes(image_path, TOWER_BBOXES)

    tower_healths = {}

    # HSV color ranges for the EMPTY part of the health bars.
    # empty ally: #425170 -> BGR(66, 81, 112) -> HSV (approx 111, 41, 44)

    empty_ally_color_range = [(np.array([100, 100, 60]), np.array([120, 255, 85]))]
    empty_ally_king_color_range = [(np.array([10, 65, 75]), np.array([20, 75, 85]))]
    empty_enemy_color_range = [(np.array([144, 88, 35]), np.array([184, 148, 95]))]
    empty_enemy_king_color_range = [(np.array([10, 120, 45]), np.array([20, 130, 55]))]

    for name, bbox in TOWER_BBOXES.items():
        x, y, w, h = bbox.to_xywh()
        health_bar_image = image[y : y + h, x : x + w]

        if "ally" in name:
            if "king" in name:
                color_range = empty_ally_king_color_range
            else:
                color_range = empty_ally_color_range
        else:  # enemy
            if "king" in name:
                color_range = empty_enemy_king_color_range
            else:
                color_range = empty_enemy_color_range

        print("calculating for " + name)
        empty_fill_ratio = _get_bar_fill_percentage(health_bar_image, color_range)

        if empty_fill_ratio is None:
            if "king_tower" in name:
                # King towers have no health bar when full. If no empty color found, assume 100%.
                health = 1.0
            else:
                # For princess towers, if bar can't be processed, assume 0 health as a fallback
                health = 0.0
            print("WARN: empty_fill_ratio is None")
        else:
            # Health is the complement of the empty ratio
            health = 1.0 - empty_fill_ratio

        tower_healths[name] = health

    return tower_healths


def is_game_active(image_path: str) -> bool:
    """
    Checks if the game is active by looking for the elixir bar (either filled or empty).
    """
    image = cv2.imread(image_path)
    if image is None:
        return False

    x, y, w, h = ELIXIR_BAR_BBOX.to_xywh()
    elixir_bar_image = image[y : y + h, x : x + w]

    # Check for purple elixir (filled bar)
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])
    purple_fill_ratio = _get_bar_fill_percentage(
        elixir_bar_image, [(lower_purple, upper_purple)]
    )
    if (
        purple_fill_ratio is not None and purple_fill_ratio > 0.1
    ):  # Threshold for purple presence
        return True

    # Check for empty elixir bar color
    lower_empty_elixir = np.array([98, 196, 34])
    upper_empty_elixir = np.array([118, 255, 134])
    empty_fill_ratio = _get_bar_fill_percentage(
        elixir_bar_image, [(lower_empty_elixir, upper_empty_elixir)]
    )
    if (
        empty_fill_ratio is not None and empty_fill_ratio > 0.5
    ):  # If more than 50% is empty color
        return True

    return False


def get_object_detections(
    screenshot_path: str, bot_screen_size: tuple[int, int]
) -> list[dict[str, Any]]:
    """
    Gets object detections from Roboflow for the given screenshot.

    WARN: This function allows the agent to continue execution without object
    detections if Roboflow encounters an error or returns no predictions.
    This behavior should be deprecated in a future, more robust version
    where object detection is critical for decision-making.
    """
    detections: list[dict[str, Any]] = []
    screen_width, screen_height = bot_screen_size

    try:
        raw_predictions = get_yolo_predictions(
            screenshot_path
        )  # Now directly returns a list of prediction dicts

        if not raw_predictions:
            print(
                "VISION WARN: No object detections received from Roboflow (or Roboflow error occurred). Proceeding without detections."
            )
            return []  # Explicitly return empty list

        for prediction in raw_predictions:
            # Ensure required keys exist
            if not all(
                k in prediction
                for k in [
                    "x",
                    "y",
                    "width",
                    "height",
                    "class",
                    "confidence",
                    "class_id",
                ]
            ):
                print(
                    f"VISION WARNING: Malformed prediction received: {prediction}. Skipping."
                )
                continue

            x_center = prediction["x"]
            y_center = prediction["y"]
            width = prediction["width"]
            height = prediction["height"]
            class_name = prediction["class"]

            # Store processed info back into detection for later use
            is_enemy = class_name.startswith("enemy-")
            base_name = class_name.removeprefix("enemy-").removeprefix("ally-")

            detections.append(
                {
                    "x": x_center,
                    "y": y_center,
                    "width": width,
                    "height": height,
                    "class": class_name,
                    "confidence": float(prediction["confidence"]),
                    "class_id": int(prediction["class_id"]),
                    "is_enemy": is_enemy,
                    "base_name": base_name,
                }
            )
    except Exception as e:
        print(
            f"VISION ERROR: An unexpected error occurred during object detection processing: {e}"
        )
        # Return empty detections if any unexpected error occurs during processing
        return []

    return detections


def get_full_game_state(
    bot: Bot,
) -> GameState:
    """
    Gets the current game state by taking a screenshot, running object detection,
    and packaging all information into a GameState object.
    """
    print("VISION: Starting get_full_game_state.")
    print("VISION: Taking screenshot...")
    screenshot_path = bot.screenshot()
    print(f"VISION: Screenshot taken: {screenshot_path}")

    if not os.path.exists(screenshot_path):
        print(f"Error: Screenshot file not found at {screenshot_path}")
        return GameState(
            elixir=0,
            tower_healths=None,
            detections=[],
            fixed_inputs=torch.zeros(FIXED_INPUT_DIM),
            card_ids=torch.empty(0, dtype=torch.long),
            card_continuous_features=torch.empty(0),
        )

    # 1. Get Elixir, Tower Healths, and Object Detections
    print("VISION: Getting elixir...")
    elixir = get_elixir(screenshot_path)
    print(f"VISION: Elixir: {elixir}")

    print("VISION: Getting tower healths...")
    tower_healths = get_tower_healths(screenshot_path)
    print(f"VISION: Tower Healths: {tower_healths}")

    screen_width, screen_height = bot.get_screen_size()
    print("VISION: Getting object detections (YOLO local inference)...")
    detections = get_object_detections(screenshot_path, (screen_width, screen_height))
    print(f"VISION: Got {len(detections)} object detections.")

    print(f"VISION: Deleting screenshot: {screenshot_path}")
    os.remove(screenshot_path)  # Clean up screenshot

    # 2. Construct fixed inputs for the agent
    ally_king_health = (
        tower_healths.get("ally-king-tower", 0.0) if tower_healths else 0.0
    )
    ally_princess_l_health = (
        tower_healths.get("ally-princess-tower-left", 0.0) if tower_healths else 0.0
    )
    ally_princess_r_health = (
        tower_healths.get("ally-princess-tower-right", 0.0) if tower_healths else 0.0
    )
    enemy_king_health = (
        tower_healths.get("enemy-king-tower", 0.0) if tower_healths else 0.0
    )
    enemy_princess_l_health = (
        tower_healths.get("enemy-princess-tower-left", 0.0) if tower_healths else 0.0
    )
    enemy_princess_r_health = (
        tower_healths.get("enemy-princess-tower-right", 0.0) if tower_healths else 0.0
    )

    ally_units_count = sum(
        1
        for d in detections
        if not d.get("is_enemy", False) and "tower" not in d.get("base_name", "")
    )
    enemy_units_count = sum(
        1
        for d in detections
        if d.get("is_enemy", False) and "tower" not in d.get("base_name", "")
    )

    fixed_inputs = torch.tensor(
        [
            elixir,
            ally_king_health,
            ally_princess_l_health,
            ally_princess_r_health,
            enemy_king_health,
            enemy_princess_l_health,
            enemy_princess_r_health,
            float(ally_units_count),
            float(enemy_units_count),
        ],
        dtype=torch.float32,
    )

    # 3. Preprocess detections into tensors
    card_ids_list, card_continuous_features_list = [], []
    if detections:
        for detection in detections:
            base_name = detection.get("base_name", "")
            try:
                if base_name in class_names:
                    card_id = class_names.index(base_name)
                    card_ids_list.append(card_id)

                    x_center = detection["x"] / screen_width
                    y_center = detection["y"] / screen_height
                    width = detection["width"] / screen_width
                    height = detection["height"] / screen_height
                    card_continuous_features_list.append(
                        [x_center, y_center, width, height]
                    )
            except (ValueError, IndexError):
                print(
                    f"Warning: Detected class base name '{base_name}' not in CLASS_NAMES. Skipping."
                )
                continue

    card_ids = (
        torch.tensor(card_ids_list, dtype=torch.long)
        if card_ids_list
        else torch.empty(0, dtype=torch.long)
    )
    card_continuous_features = (
        torch.tensor(card_continuous_features_list, dtype=torch.float32)
        if card_continuous_features_list
        else torch.empty(0)
    )

    # 4. Create and return GameState object
    print("VISION: Finished get_full_game_state.")
    return GameState(
        elixir=elixir,
        tower_healths=tower_healths,
        detections=detections,
        fixed_inputs=fixed_inputs,
        card_ids=card_ids,
        card_continuous_features=card_continuous_features,
    )


def calculate_reward(
    current_state: GameState,
    previous_state: GameState,
) -> float:
    """
    Calculates the reward based on the change between two game states.

    The reward is based on:
    - Damage dealt to enemy towers.
    - Damage taken by ally towers.
    - Change in the number of ally and enemy units.
    - Change in elixir.
    """
    reward = 0.0

    # --- 1. Tower Health Rewards/Penalties ---
    if current_state.tower_healths and previous_state.tower_healths:
        for tower_name, current_health in current_state.tower_healths.items():
            previous_health = previous_state.tower_healths.get(
                tower_name, current_health
            )
            health_delta = current_health - previous_health
            if "enemy" in tower_name and health_delta < 0:
                reward += (
                    abs(health_delta) * 15
                )  # Increased reward for damaging enemy towers
            elif "ally" in tower_name and health_delta < 0:
                reward -= (
                    abs(health_delta) * 15
                )  # Increased penalty for ally towers taking damage

    # --- 2. Unit Change Rewards/Penalties ---
    current_ally_units = {
        d["class"] for d in current_state.detections if not d.get("is_enemy")
    }
    previous_ally_units = {
        d["class"] for d in previous_state.detections if not d.get("is_enemy")
    }
    current_enemy_units = {
        d["class"] for d in current_state.detections if d.get("is_enemy")
    }
    previous_enemy_units = {
        d["class"] for d in previous_state.detections if d.get("is_enemy")
    }

    # Reward for destroying enemy units
    destroyed_enemies = len(previous_enemy_units - current_enemy_units)
    reward += destroyed_enemies * 2.0

    # Penalty for losing ally units
    lost_allies = len(previous_ally_units - current_ally_units)
    reward -= lost_allies * 2.0

    # --- 3. Elixir Advantage Reward ---
    elixir_advantage = current_state.elixir - previous_state.elixir
    if elixir_advantage > 0:
        reward += elixir_advantage * 0.1  # Small reward for gaining elixir

    # --- 4. Elixir Overflow Penalty ---
    if current_state.elixir == 10:
        penalty = -5.0
        reward += penalty
        print(f"Calculated reward: Elixir overflow penalty applied: {penalty}")

    # --- 5. Win/Loss Condition ---
    if current_state.tower_healths:
        if current_state.tower_healths.get("enemy_king_tower", 1.0) <= 0.01:
            reward += 100  # Big reward for winning
        if current_state.tower_healths.get("ally_king_tower", 1.0) <= 0.01:
            reward -= 100  # Big penalty for losing

    if reward != 0.0:
        print(f"Calculated reward: {reward}")
    else:
        print("Calculated reward: 0.0 (No significant change in state)")

    return reward
