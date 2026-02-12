from typing import Optional, Any
import cv2
import numpy as np
import os
import torch

from roboflow_service import get_roboflow_predictions
from positions import ELIXIR_BAR_BBOX, BBox
from game_state import GameState
from bot import Bot
from cards import CARDS, TOWERS

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

def _get_bar_fill_percentage(bar_image: np.ndarray, color_ranges: list[tuple[np.ndarray, np.ndarray]]) -> Optional[float]:
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
    elixir_bar_image = image[y:y+h, x:x+w]

    # Purple color range in HSV
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])

    fill_ratio = _get_bar_fill_percentage(elixir_bar_image, [(lower_purple, upper_purple)])

    if fill_ratio is None:
        return 0
        
    # Convert the fill ratio to an elixir value (0-10)
    elixir = int(round(fill_ratio * 10))

    return elixir

def get_tower_healths(image_path: str) -> dict[str, float]:
    """
    Calculates the health of each tower from a screenshot of the game.
    Returns a dictionary with tower names as keys and health (0.0-1.0) as values.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {}

    tower_healths = {}

    # Bounding Boxes for each tower's health bar, using BBox(x1, y1, x2, y2)
    tower_bboxes = {
        "ally_king_tower":          BBox(480, 1450, 600, 1460),
        "ally_left_princess_tower": BBox(420, 1070, 530, 1090),
        "ally_right_princess_tower":BBox(930, 1070, 1040, 1090),
        "enemy_king_tower":         BBox(480, 350, 600, 360),
        "enemy_left_princess_tower":BBox(420, 190, 530, 210),
        "enemy_right_princess_tower":BBox(930, 190, 1040, 210),
    }

    # HSV color ranges for the EMPTY part of the health bars, with wider S and V ranges for robustness
    # empty ally: #2d384f -> BGR(79, 56, 45) -> HSV(108, 110, 79)
    empty_ally_color_range = [
        (np.array([100, 50, 40]), np.array([115, 170, 140]))
    ]
    # empty enemy: #412333 -> BGR(51, 35, 65) -> HSV(164, 117, 64)
    empty_enemy_color_range = [
        (np.array([155, 60, 30]), np.array([175, 180, 120]))
    ]
    
    for name, bbox in tower_bboxes.items():
        x, y, w, h = bbox.to_xywh()
        health_bar_image = image[y:y+h, x:x+w]
        
        if "ally" in name:
            color_range = empty_ally_color_range
        else: # enemy
            color_range = empty_enemy_color_range
            
        empty_fill_ratio = _get_bar_fill_percentage(health_bar_image, color_range)
            
        if empty_fill_ratio is None:
            # If the bar can't be processed, assume 0 health as a fallback
            health = 0.0
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
    elixir_bar_image = image[y:y+h, x:x+w]

    # Check for purple elixir (filled bar)
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])
    purple_fill_ratio = _get_bar_fill_percentage(elixir_bar_image, [(lower_purple, upper_purple)])
    if purple_fill_ratio is not None and purple_fill_ratio > 0.1: # Threshold for purple presence
        return True

    # Check for empty elixir bar color
    lower_empty_elixir = np.array([98, 196, 34])
    upper_empty_elixir = np.array([118, 255, 134])
    empty_fill_ratio = _get_bar_fill_percentage(elixir_bar_image, [(lower_empty_elixir, upper_empty_elixir)])
    if empty_fill_ratio is not None and empty_fill_ratio > 0.5: # If more than 50% is empty color
        return True

    return False

def get_object_detections(screenshot_path: str, bot_screen_size: tuple[int, int]) -> list[dict[str, Any]]:
    """
    Gets object detections from Roboflow for the given screenshot.
    """
    detections: list[dict[str, Any]] = []
    try:
        roboflow_response = get_roboflow_predictions(screenshot_path)
        
        # Extract the Detections object
        roboflow_detections_obj = roboflow_response.get("predictions")

        if roboflow_detections_obj is not None:
            screen_width, screen_height = bot_screen_size
            for i in range(len(roboflow_detections_obj.xyxy)):
                x1, y1, x2, y2 = roboflow_detections_obj.xyxy[i]
                
                # Calculate center x, y, width, height
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Ensure 'class_name' exists in data
                class_name = "unknown"
                if "class_name" in roboflow_detections_obj.data and i < len(roboflow_detections_obj.data["class_name"]):
                    class_name = roboflow_detections_obj.data["class_name"][i]

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
                        "confidence": float(roboflow_detections_obj.confidence[i]),
                        "class_id": int(roboflow_detections_obj.class_id[i]),
                        "is_enemy": is_enemy,
                        "base_name": base_name,
                    }
                )
    except Exception as e:
        print(f"Error during Roboflow object detection: {e}")
        # Return empty detections if an error occurs
        return [] 
    
    return detections

def get_full_game_state(
    bot: Bot,
) -> GameState:
    """
    Gets the current game state by taking a screenshot, running object detection,
    and packaging all information into a GameState object.
    """
    screenshot_path = bot.screenshot()
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
    elixir = get_elixir(screenshot_path)
    tower_healths = get_tower_healths(screenshot_path)
    screen_width, screen_height = bot.get_screen_size()
    detections = get_object_detections(screenshot_path, (screen_width, screen_height))
    
    os.remove(screenshot_path)  # Clean up screenshot

    # 2. Construct fixed inputs for the agent
    ally_king_health = tower_healths.get("ally-king-tower", 0.0) if tower_healths else 0.0
    ally_princess_l_health = tower_healths.get("ally-princess-tower-left", 0.0) if tower_healths else 0.0
    ally_princess_r_health = tower_healths.get("ally-princess-tower-right", 0.0) if tower_healths else 0.0
    enemy_king_health = tower_healths.get("enemy-king-tower", 0.0) if tower_healths else 0.0
    enemy_princess_l_health = tower_healths.get("enemy-princess-tower-left", 0.0) if tower_healths else 0.0
    enemy_princess_r_health = tower_healths.get("enemy-princess-tower-right", 0.0) if tower_healths else 0.0

    ally_units_count = sum(1 for d in detections if not d.get("is_enemy", False) and "tower" not in d.get("base_name", ""))
    enemy_units_count = sum(1 for d in detections if d.get("is_enemy", False) and "tower" not in d.get("base_name", ""))

    fixed_inputs = torch.tensor([
        elixir,
        ally_king_health, ally_princess_l_health, ally_princess_r_health,
        enemy_king_health, enemy_princess_l_health, enemy_princess_r_health,
        float(ally_units_count), float(enemy_units_count),
    ], dtype=torch.float32)

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
                    card_continuous_features_list.append([x_center, y_center, width, height])
            except (ValueError, IndexError):
                print(f"Warning: Detected class base name '{base_name}' not in CLASS_NAMES. Skipping.")
                continue

    card_ids = torch.tensor(card_ids_list, dtype=torch.long) if card_ids_list else torch.empty(0, dtype=torch.long)
    card_continuous_features = torch.tensor(card_continuous_features_list, dtype=torch.float32) if card_continuous_features_list else torch.empty(0)

    # 4. Create and return GameState object
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
            previous_health = previous_state.tower_healths.get(tower_name, current_health)
            health_delta = current_health - previous_health
            if "enemy" in tower_name and health_delta < 0:
                reward += abs(health_delta) * 15  # Increased reward for damaging enemy towers
            elif "ally" in tower_name and health_delta < 0:
                reward -= abs(health_delta) * 15  # Increased penalty for ally towers taking damage

    # --- 2. Unit Change Rewards/Penalties ---
    current_ally_units = {d['class'] for d in current_state.detections if not d.get('is_enemy')}
    previous_ally_units = {d['class'] for d in previous_state.detections if not d.get('is_enemy')}
    current_enemy_units = {d['class'] for d in current_state.detections if d.get('is_enemy')}
    previous_enemy_units = {d['class'] for d in previous_state.detections if d.get('is_enemy')}

    # Reward for destroying enemy units
    destroyed_enemies = len(previous_enemy_units - current_enemy_units)
    reward += destroyed_enemies * 2.0

    # Penalty for losing ally units
    lost_allies = len(previous_ally_units - current_ally_units)
    reward -= lost_allies * 2.0

    # --- 3. Elixir Advantage Reward ---
    elixir_advantage = current_state.elixir - previous_state.elixir
    if elixir_advantage > 0:
        reward += elixir_advantage * 0.1 # Small reward for gaining elixir

    # --- 4. Win/Loss Condition ---
    if current_state.tower_healths:
        if current_state.tower_healths.get("enemy_king_tower", 1.0) <= 0.01:
            reward += 100 # Big reward for winning
        if current_state.tower_healths.get("ally_king_tower", 1.0) <= 0.01:
            reward -= 100 # Big penalty for losing
            
    if reward != 0.0:
        print(f"Calculated reward: {reward}")
    else:
        print("Calculated reward: 0.0 (No significant change in state)")

    return reward