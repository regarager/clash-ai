import cv2
import numpy as np

def get_elixir(image_path: str) -> int:
    """
    Calculates the current elixir count from a screenshot of the game.
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0

    # Bounding box for the elixir bar [x, y, w, h]
    # These values might need adjustment depending on the device's screen resolution.
    bbox = (290, 1775, 490, 30)
    x, y, w, h = bbox
    elixir_bar_image = image[y:y+h, x:x+w]

    hsv_image = cv2.cvtColor(elixir_bar_image, cv2.COLOR_BGR2HSV)

    # Purple color range in HSV
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])

    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # Calculate the percentage of the bar that is purple
    purple_pixels = cv2.countNonZero(mask)
    total_pixels = w * h
    if total_pixels == 0:
        return 0
        
    fill_ratio = purple_pixels / total_pixels

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

    # Bounding Boxes (x, y, w, h) for each tower's health bar
    tower_bboxes = {
        "ally_king_tower": (480, 1450, 120, 10),
        "ally_left_princess_tower": (290, 1250, 120, 10),
        "ally_right_princess_tower": (670, 1250, 120, 10),
        "enemy_king_tower": (480, 350, 120, 10),
        "enemy_left_princess_tower": (290, 550, 120, 10),
        "enemy_right_princess_tower": (670, 550, 120, 10),
    }

    # Color ranges for health bars (ally=blue, enemy=red)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    for name, (x, y, w, h) in tower_bboxes.items():
        health_bar_image = image[y:y+h, x:x+w]
        hsv_image = cv2.cvtColor(health_bar_image, cv2.COLOR_BGR2HSV)
        
        if "ally" in name:
            mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        else: # enemy
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
        health_pixels = cv2.countNonZero(mask)
        total_pixels = w * h
        if total_pixels == 0:
            health_ratio = 0.0
        else:
            health_ratio = health_pixels / total_pixels
        
        tower_healths[name] = health_ratio
        
    return tower_healths
