import os
import dotenv
from roboflow import Roboflow

def get_api_key():
    """
    Checks for the Roboflow API key in environment variables or a .env file.
    If not found, prompts the user and saves it to a .env file.
    """
    dotenv.load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        print("--- Roboflow API Key Setup ---")
        print("Your Roboflow API key is required to download the object detection model.")
        print("You can find your key here: https://universe.roboflow.com/settings/api")
        api_key = input("Please enter your Roboflow API key: ").strip()

        if not api_key:
            print("ERROR: API key cannot be empty.")
            exit()

        with open(".env", "w") as f:
            f.write(f"ROBOFLOW_API_KEY={api_key}\n")
        print("Successfully saved API key to .env file.")
        # Load the newly created .env file
        dotenv.load_dotenv()
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    return api_key

def download_model(api_key):
    """
    Downloads the YOLOv8 model from Roboflow.
    """
    model_path = "clash-ai-8/best.pt"

    if os.path.exists(model_path):
        print(f"Local model already found at '{model_path}'. Skipping download.")
        return

    print("Downloading model from Roboflow...")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("stuff-m0fm7").project("clash-ai-kimrx")
        version = project.version(8)
        version.download("yolov8")
        print("Model downloaded successfully.")
        print("The model is saved in the 'clash-ai-8' directory.")
    except Exception as e:
        print(f"Error downloading model from Roboflow: {e}")
        print("\nPlease check your API key and project path.")
        exit()

def main():
    """
    Main setup script.
    """
    print("--- Clash Royale AI Bot Setup ---")
    api_key = get_api_key()
    download_model(api_key)
    print("\nSetup complete.")

if __name__ == "__main__":
    main()