import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient # Import InferenceHTTPClient
import time # Keep for rate limiting
from typing import Any # Import Any for type hinting

load_dotenv() # Load environment variables from .env file

# Retrieve API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("ROBOFLOW: ROBOFLOW_API_KEY not found in environment variables.")
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please set it in a .env file.")
else:
    print(f"ROBOFLOW: Using API Key (first 5 chars): {api_key[:5]}*****")

# Initialize the Roboflow InferenceHTTPClient globally
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Model ID to be used for inference
MODEL_ID = "clash-ai-kimrx/10"

# Rate limiting variables
_last_roboflow_request_time = 0.0
MIN_ROBOFLOW_REQUEST_INTERVAL = 2.0 # seconds between Roboflow API calls

def _respect_roboflow_rate_limit():
    global _last_roboflow_request_time
    time_since_last_request = time.monotonic() - _last_roboflow_request_time

    if time_since_last_request < MIN_ROBOFLOW_REQUEST_INTERVAL:
        sleep_duration = MIN_ROBOFLOW_REQUEST_INTERVAL - time_since_last_request
        print(f"ROBOFLOW: Rate limiting - waiting for {sleep_duration:.2f} seconds.")
        time.sleep(sleep_duration)
    
    _last_roboflow_request_time = time.monotonic()

def get_roboflow_predictions(image_path: str, display_image: bool = False) -> list[dict[str, Any]]: # Change return type hint
    _respect_roboflow_rate_limit() # Enforce rate limit before making the request

    print(f"ROBOFLOW: Making request with model {MODEL_ID} for image: {image_path}")

    try:
        result = CLIENT.infer(image_path, model_id=MODEL_ID)

        if result and "predictions" in result: # Check if result is not empty and has a 'predictions' key
            actual_predictions = result["predictions"]
            print(f"ROBOFLOW: Request successful. Got {len(actual_predictions)} predictions.")
            return actual_predictions # <--- Return the list of actual predictions
        else:
            print(f"ROBOFLOW WARNING: No 'predictions' key or empty predictions returned from CLIENT.infer(). Result: {result}")
            return [] # <--- Return empty list directly
    except Exception as e:
        print(f"ROBOFLOW ERROR: An error occurred during Roboflow inference: {e}")
        return [] # <--- Return empty list on error
if __name__ == "__main__":
    # Example usage (for testing this module directly)
    print("This module is intended to be imported as a service.")
    print("To test, you would typically call get_roboflow_predictions with an image path.")
    # Example:
    # try:
    #     test_image_path = "path/to/your/test_image.jpg"
    #     if os.path.exists(test_image_path):
    #         print(f"Testing with {test_image_path}...")
    #         results = get_roboflow_predictions(test_image_path, display_image=True)
    #         print("Test predictions:", results)
    #         cv2.destroyAllWindows()
    #     else:
    #         print(f"Test image not found at {test_image_path}")
    # except Exception as e:
    #     print(f"An error occurred during testing: {e}")
