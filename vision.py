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
