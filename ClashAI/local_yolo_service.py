import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
)  # Import List and Dict for clearer type hints

from PIL import Image  # For loading images
from ultralytics import YOLO  # For loading and running YOLO models

from ClashAI.logger import *

# Path to the local YOLOv11 model
YOLO_MODEL_PATH = "vision/best.pt"

# Lazy-loaded YOLO model instance
_yolo_model: Optional[YOLO] = None

# Rate limiting variables (retained for consistency, though local inference is fast)
_last_yolo_request_time = 0.0
MIN_YOLO_REQUEST_INTERVAL = (
    0.5  # seconds between YOLO inference calls (can be faster than API)
)


def _respect_yolo_rate_limit():
    global _last_yolo_request_time
    time_since_last_request = time.monotonic() - _last_yolo_request_time

    if time_since_last_request < MIN_YOLO_REQUEST_INTERVAL:
        sleep_duration = MIN_YOLO_REQUEST_INTERVAL - time_since_last_request
        warn(f"YOLO: Rate limiting - waiting for {sleep_duration:.2f} seconds.")
        time.sleep(sleep_duration)

    _last_yolo_request_time = time.monotonic()


def get_yolo_predictions(image_path: str) -> List[Dict[str, Any]]:
    """
    Performs object detection using a local YOLOv11 model on a given image.

    Args:
        image_path (str): The path to the image file to analyze.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a detected object with its bounding box,
                              class, confidence, and class ID.
    """
    global _yolo_model
    predictions_list: List[Dict[str, Any]] = []
    _respect_yolo_rate_limit()  # Enforce rate limit before making the request

    if _yolo_model is None:
        try:
            info(f"YOLO: Loading model from {YOLO_MODEL_PATH}...")
            _yolo_model = YOLO(YOLO_MODEL_PATH)
            info("YOLO: Model loaded successfully.")
        except Exception as e:
            error(f"YOLO ERROR: Failed to load model from {YOLO_MODEL_PATH}: {e}")
            return []  # Return empty predictions on model load failure

    info(f"YOLO: Performing local inference for image: {image_path}")
    try:
        pil_image = Image.open(image_path)

        # Perform inference
        # The .predict() method returns a list of Results objects
        results_list = _yolo_model.predict(source=pil_image, verbose=False)

        if results_list:
            # Assuming single image inference, so take the first Results object
            first_result = results_list[0]

            if first_result.boxes is not None:
                for box in first_result.boxes:
                    # box.xywh gives [x_center, y_center, width, height] in pixels
                    x_center, y_center, width, height = box.xywh[0].tolist()

                    predictions_list.append(
                        {
                            "x": x_center,
                            "y": y_center,
                            "width": width,
                            "height": height,
                            "class": first_result.names[
                                int(box.cls[0])
                            ],  # Get class name from model.names
                            "confidence": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                        }
                    )
            info(
                f"YOLO: Local inference complete. Got {len(predictions_list)} predictions."
            )
        else:
            warn("YOLO WARNING: No inference results returned from local model.")

    except Exception as e:
        error(
            f"YOLO ERROR: An unexpected error occurred during local YOLO inference: {e}"
        )

    return predictions_list


if __name__ == "__main__":
    # Example usage (for testing this module directly)
    warn("This module is intended to be imported as a service.")
    warn("To test, you would typically call get_yolo_predictions with an image path.")
    # Example:
    # try:
    #     test_image_path = "path/to/your/test_image.jpg"
    #     if os.path.exists(test_image_path):
    #         print(f"Testing with {test_image_path}...")
    #         results = get_yolo_predictions(test_image_path)
    #         print("Test predictions:", results)
    #     else:
    #         print(f"Test image not found at {test_image_path}")
    # except Exception as e:
    #     print(f"An error occurred during testing: {e}")
