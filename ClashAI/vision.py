from enum import Enum, auto
import os
import time  # For timestamp in debug image filename
from typing import Any, Optional

import cv2
import numpy as np
import torch

from .bot import Bot
from .cards import CARDS, TOWERS
from .game_state import GameScreen, GameState
from .local_yolo_service import get_yolo_predictions
from .positions import ELIXIR_BAR_BBOX, END_SCREEN_BBOX, MAIN_PAGE_BBOX, TOWER_BBOXES, BBox
from .hand_reader import HandReader

# Global instance of HandReader
_hand_reader = HandReader()


def reset_hand_reader():
    """Resets the active deck in HandReader. Call this at the start of a match."""
    _hand_reader.reset_active_deck()


# --- Model and State Configuration Constants (moved from main.py) ---
FIXED_INPUT_DIM = 9  # Elixir, 6 Tower Healths, Ally Unit Count, Enemy Unit Count


def get_base_name(name: str) -> str:
    return name.removeprefix("enemy-").removeprefix("ally-")


# The following are derived from the downloaded model's data.yaml
CLASS_NAMES = sorted(
    list(
        set(
            [get_base_name(card) for card in CARDS]
            + [get_base_name(tower) for tower in TOWERS]
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


def _debug_save_hand_crops(image_path: str, card_slots: list[tuple[int, int]], crop_w: int, crop_h: int):
    """
    Visualizes the card slots on the screenshot to verify positioning.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return

        for i, (x_center, y_center) in enumerate(card_slots):
            x1 = x_center - crop_w // 2
            y1 = y_center - crop_h // 2
            x2 = x_center + crop_w // 2
            y2 = y_center + crop_h // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(img, f"Slot {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        debug_path = os.path.join(os.path.dirname(image_path), f"debug_hand_slots_{int(time.time())}.png")
        cv2.imwrite(debug_path, img)
        print(f"DEBUG: Saved hand slot visualization to {debug_path}")
    except Exception as e:
        print(f"DEBUG ERROR: Failed to save hand slot visualization: {e}")


def get_tower_healths(image_path: str) -> dict[str, float]:
    """
    Calculates the health of each tower from a screenshot of the game.
    Returns a dictionary with tower names as keys and health (0.0-1.0) as values.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {}

    # _debug_save_health_bboxes(image_path, TOWER_BBOXES)

    tower_healths = {}

    # HSV color ranges for the EMPTY part of the health bars.
    # empty ally: #425170 -> BGR(66, 81, 112) -> HSV (approx 111, 41, 44)

    TOLERANCE = 5
    ally_color = np.array([99, 140, 255])
    ally_color_range = [(ally_color - TOLERANCE, ally_color + TOLERANCE)]

    enemy_color = np.array([171, 209, 229])
    enemy_color_range = [(enemy_color - TOLERANCE, enemy_color + TOLERANCE)]

    ally_king_empty_color = np.array([13, 71, 112])
    enemy_king_empty_color = np.array([13, 127, 76])
    ally_king_color_range = [
        (ally_king_empty_color - TOLERANCE, ally_king_empty_color + TOLERANCE)
    ]
    enemy_king_color_range = [
        (enemy_king_empty_color - TOLERANCE, enemy_king_empty_color + TOLERANCE)
    ]

    for name, bbox in TOWER_BBOXES.items():
        x, y, w, h = bbox.to_xywh()
        health_bar_image = image[y : y + h, x : x + w]

        print("calculating for " + name)

        # debug_color_range(image_path, bbox, name)

        if "king" in name:
            if "ally" in name:
                color_range = ally_king_color_range
            else:
                color_range = enemy_king_color_range
            empty_fill_ratio = _get_bar_fill_percentage(health_bar_image, color_range)

            tower_healths[name] = 1.0 - (empty_fill_ratio or 0.0)
        else:
            if "ally" in name:
                color_range = ally_color_range
            else:  # enemy
                color_range = enemy_color_range

            fill_ratio = _get_bar_fill_percentage(health_bar_image, color_range)

            tower_healths[name] = fill_ratio

    return tower_healths


def get_game_screen(image_path: str) -> GameScreen:
    """
    Detects the current game screen from a screenshot.
    """
    image = cv2.imread(image_path)
    if image is None:
        return GameScreen.UNKNOWN

    # 1. Check for Active Battle Screen (GAME_SCREEN)
    # Primary indicator: presence of the elixir bar (purple or background color)
    x, y, w, h = ELIXIR_BAR_BBOX.to_xywh()
    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
        elixir_bar_image = image[y : y + h, x : x + w]

        # Purple color range (filled bar)
        lower_purple = np.array([125, 50, 50])
        upper_purple = np.array([155, 255, 255])
        purple_fill = _get_bar_fill_percentage(
            elixir_bar_image, [(lower_purple, upper_purple)]
        )

        # Empty elixir bar color
        lower_empty = np.array([98, 196, 34])
        upper_empty = np.array([118, 255, 134])
        empty_fill = _get_bar_fill_percentage(
            elixir_bar_image, [(lower_empty, upper_empty)]
        )

        if (purple_fill is not None and purple_fill > 0.05) or (
            empty_fill is not None and empty_fill > 0.3
        ):
            return GameScreen.GAME_SCREEN

    # 2. Check for End Screen (Victory/Defeat)
    # Blue: #4983b2 -> RGB(73, 131, 178) -> HSV approx (104, 150, 178)
    ex, ey, ew, eh = END_SCREEN_BBOX.to_xywh()
    if 0 <= ey < image.shape[0] and 0 <= ex < image.shape[1]:
        end_screen_region = image[ey : ey + eh, ex : ex + ew]

        lower_blue = np.array([95, 100, 130])
        upper_blue = np.array([115, 255, 220])
        blue_fill = _get_bar_fill_percentage(
            end_screen_region, [(lower_blue, upper_blue)]
        )

        if blue_fill is not None and blue_fill >= 0.75:
            return GameScreen.END_SCREEN
    
    # 3. Check for Main Page
    # Yellow: #ffbb00 -> RGB(255, 187, 0) -> HSV approx (22, 255, 255)
    mx, my, mw, mh = MAIN_PAGE_BBOX.to_xywh()
    if 0 <= my < image.shape[0] and 0 <= mx < image.shape[1]:
        main_page_region = image[my : my + mh, mx : mx + mw]

        lower_yellow = np.array([15, 150, 150])
        upper_yellow = np.array([30, 255, 255])
        yellow_fill = _get_bar_fill_percentage(
            main_page_region, [(lower_yellow, upper_yellow)]
        )

        if yellow_fill is not None and yellow_fill >= 0.75:
            return GameScreen.MAIN_PAGE

    return GameScreen.UNKNOWN


def is_game_active(image_path: str) -> bool:
    """
    Checks if the game is in the active battle screen.
    """
    return get_game_screen(image_path) == GameScreen.GAME_SCREEN


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
            card_ids=torch.zeros(4, dtype=torch.long),
            card_continuous_features=torch.empty(0),
            playable_mask=torch.zeros(4, dtype=torch.bool),
            screen_type=GameScreen.UNKNOWN,
        )

    # 0. Detect Screen Type
    screen_type = get_game_screen(screenshot_path)
    print(f"VISION: Detected Screen: {screen_type.name}")

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

    # 1.5 Get Hand Cards (Precise Identification)
    image = cv2.imread(screenshot_path)
    hand_info = _hand_reader.identify_hand(image)
    print(f"VISION: Hand cards identified: {[h['name'] for h in hand_info]}")

    # Debug: Verify positions
    _debug_save_hand_crops(screenshot_path, _hand_reader.card_slots, _hand_reader.crop_w, _hand_reader.crop_h)

    # Optional: Save crops periodically to build training data
    _hand_reader.save_hand_crops(image)

    # Reclassify GAME_SCREEN as UNKNOWN if no objects are detected
    if screen_type == GameScreen.GAME_SCREEN and not detections:
        print("VISION: GAME_SCREEN detected but zero objects found. Reclassifying as UNKNOWN.")
        screen_type = GameScreen.UNKNOWN

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

    # 3. Preprocess Hand Cards into card_ids (Fixed size: 4)
    # 0 = unknown, 1..8 = templates
    card_ids_list = []
    playable_list = []
    
    # Import whitelisted names from hand_reader
    from .hand_reader import ALLOWED_TEMPLATES
    whitelist = sorted(list(ALLOWED_TEMPLATES))
    
    for i, h in enumerate(hand_info):
        name = h["name"]
        playable_list.append(h["playable"])
        
        if name in whitelist:
            # Map to 1..8 based on sorted whitelist
            card_id = whitelist.index(name) + 1
        else:
            card_id = 0
            
        card_ids_list.append(card_id)

    card_ids = torch.tensor(card_ids_list, dtype=torch.long)
    print(f"VISION: Hand card IDs: {card_ids_list} (Names: {[h['name'] for h in hand_info]})")
    playable_mask = torch.tensor(playable_list, dtype=torch.bool)

    # 4. Preprocess YOLO Detections into continuous features
    # Note: We filter out detections that are likely hand cards to focus on field state
    field_detections = [
        d for d in detections 
        if d["y"] < 1400 # Heuristic: anything above the hand area is a field unit
    ]
    
    card_continuous_features_list = []
    for detection in field_detections:
        x_center = detection["x"] / screen_width
        y_center = detection["y"] / screen_height
        width = detection["width"] / screen_width
        height = detection["height"] / screen_height
        card_continuous_features_list.append(
            [x_center, y_center, width, height]
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
        playable_mask=playable_mask,
        screen_type=screen_type,
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

    if current_state.tower_healths and previous_state.tower_healths:
        prev_difference = 0
        current_difference = 0

        for tower_name, current_health in current_state.tower_healths.items():
            if "king" in tower_name:
                continue

            if "enemy" in tower_name:
                prev_difference -= previous_state.tower_healths.get(
                    tower_name, current_health
                )
                current_difference -= current_health
            else:
                prev_difference += previous_state.tower_healths.get(
                    tower_name, current_health
                )
                current_difference += current_health

        reward += 100 * (current_difference - prev_difference)

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
    reward += destroyed_enemies * 5.0

    # Penalty for losing ally units
    lost_allies = len(previous_ally_units - current_ally_units)
    reward -= lost_allies * 5.0

    # --- 4. Elixir Overflow Penalty ---
    if current_state.elixir >= 8:
        penalty = 50.0
        reward -= penalty
        print(f"Calculated reward: Elixir overflow penalty applied: {penalty}")

    # --- 5. Win/Loss Condition ---
    if current_state.tower_healths:
        if current_state.tower_healths.get("enemy_king_tower", 1.0) <= 0.01:
            reward += 500  # Big reward for winning
        if current_state.tower_healths.get("ally_king_tower", 1.0) <= 0.01:
            reward -= 500  # Big penalty for losing

    if reward != 0.0:
        print(f"Calculated reward: {reward}")
    else:
        print("Calculated reward: 0.0 (No significant change in state)")

    return reward
