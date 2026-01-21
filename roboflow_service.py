import os
from dotenv import load_dotenv
from inference import InferencePipeline
import cv2

load_dotenv() # Load environment variables from .env file

# Retrieve API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please set it in a .env file.")

# Global pipeline instance (can be initialized once if preferred for performance)
# Or re-initialized per call if input type changes or for isolated calls
_pipeline_instance = None

def get_roboflow_predictions(image_path: str, display_image: bool = False) -> dict:
    """
    Sends an image to the Roboflow Inference API and returns the predictions.

    Args:
        image_path (str): The path to the image file to analyze.
        display_image (bool): Whether to display the workflow image using OpenCV.

    Returns:
        dict: The prediction results from the Roboflow API.
    """
    global _pipeline_instance
    predictions = {}

    def _prediction_sink(result, video_frame):
        nonlocal predictions
        predictions = result
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
        pipeline.join() # Wait for the pipeline to finish processing the single image
    except Exception as e:
        print(f"Error during Roboflow inference: {e}")
    finally:
        # It's good practice to ensure resources are released if the pipeline is short-lived
        # For a single image, the pipeline should naturally close after processing
        pass 
    
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