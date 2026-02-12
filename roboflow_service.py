import os
from dotenv import load_dotenv
from inference import InferencePipeline
import cv2
import time # Import time for sleep

load_dotenv() # Load environment variables from .env file

# Retrieve API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please set it in a .env file.")

# Global pipeline instance (can be initialized once if preferred for performance)
# Or re-initialized per call if input type changes or for isolated calls
_pipeline_instance = None

# Flag to signal that predictions have been received
prediction_received = False

def get_roboflow_predictions(image_path: str, display_image: bool = False) -> dict:
    """
    Sends an image to the Roboflow Inference API and returns the predictions.

    Args:
        image_path (str): The path to the image file to analyze.
        display_image (bool): Whether to display the workflow image using OpenCV.

    Returns:
        dict: The prediction results from the Roboflow API.
    """
    global prediction_received # Declare as global here
    predictions = {}
    prediction_received = False # Reset flag for each call

    def _prediction_sink(result, video_frame):
        nonlocal predictions
        global prediction_received # Corrected: use global for module-level variable
        predictions = result["predictions"]
        prediction_received = True # Signal that predictions have been received
        if display_image and result.get("output_image"):
            cv2.imshow("Workflow Image", result["output_image"].numpy_image)
            cv2.waitKey(1)

    # Initialize pipeline for image processing (setting video_reference to image_path)
    # Re-initialize each time to handle different image paths effectively
    pipeline = InferencePipeline.init_with_workflow(
        api_key=api_key,
        workspace_name="stuff-m0fm7",
        workflow_id="custom-workflow",
        video_reference=image_path, # Process a single image file
        max_fps=1, # Only one frame (the image) will be processed
        on_prediction=_prediction_sink
    )

    try:
        pipeline.start()
        # Wait until predictions are received, then stop the pipeline
        while not prediction_received:
            time.sleep(0.1) # Check frequently

        pipeline.stop() # Explicitly stop the pipeline once predictions are received
        pipeline.join() # Ensure all threads are cleaned up
    except Exception as e:
        print(f"Error during Roboflow inference: {e}")
    finally:
        pass # The pipeline is now explicitly stopped and joined

    return predictions

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
