import torch
import os
import yaml
from time import sleep
from roboflow import Roboflow
from ultralytics import YOLO
import dotenv # Import the dotenv library

# Load environment variables from .env file at the very beginning
dotenv.load_dotenv()

from adb_pywrapper.adb_device import AdbDevice
from bot import Bot
from agent import ActorCritic

# --- Model and State Configuration ---
FIXED_INPUT_DIM = 3
# The following are derived from the downloaded model's data.yaml
NUM_CARD_TYPES = None 
CLASS_NAMES = None
CARD_CONTINUOUS_DIM = 4      # [center_x, center_y, width, height] from YOLO output
NUM_DISCRETE_ACTIONS = 5
CONTINUOUS_ACTION_DIM = 2
CARD_EMBEDDING_SIZE = 16

def get_state(bot: Bot, local_model: YOLO):
    """
    Gets the current game state by taking a screenshot and running the local
    YOLOv8 object detection model.
    """
    print("Capturing screenshot and running local detection...")
    screenshot_path = bot.screenshot()
    if not os.path.exists(screenshot_path):
        print(f"Error: Screenshot file not found at {screenshot_path}")
        return torch.randn(FIXED_INPUT_DIM), torch.empty(0, dtype=torch.long), torch.empty(0)

    # 1. Run local object detection model
    try:
        results = local_model(screenshot_path, verbose=False)
        # Assumes results[0] is the primary result object
        predictions = results[0].boxes
    except Exception as e:
        print(f"Error during local model prediction: {e}")
        return torch.randn(FIXED_INPUT_DIM), torch.empty(0, dtype=torch.long), torch.empty(0)
    finally:
        os.remove(screenshot_path)

    print(f"Detected {len(predictions)} objects.")

    # 2. Simulate fixed inputs (placeholder)
    fixed_inputs = torch.randn(FIXED_INPUT_DIM)

    # 3. Preprocess detections into tensors
    if len(predictions) == 0:
        return fixed_inputs, torch.empty(0, dtype=torch.long), torch.empty(0)

    # Ultralytics provides tensors directly, which is efficient
    card_ids = predictions.cls.long() # Class IDs are already integers
    # Get normalized xywh bounding boxes
    card_continuous_features = predictions.xywhn 

    return fixed_inputs, card_ids, card_continuous_features


def main():
    """
    Main loop for the Clash Royale AI bot using a local YOLO model.
    """
    print("--- Clash Royale AI Bot (Local Inference) ---")
    print("SECURITY REMINDER: Do not share your API key. Use the ROBOFLOW_API_KEY environment variable.")

    # --- Download Roboflow Model (if needed) ---
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("\nERROR: ROBOFLOW_API_KEY is required to download the model for the first time.")
        print("Please create a .env file in the project root with ROBOFLOW_API_KEY='YOUR_API_KEY'")
        print("Or set it as an environment variable before running.")
        exit()

    model_path = "clash-ai-kimrx-instant-9/best.pt"
    yaml_path = "clash-ai-kimrx-instant-9/data.yaml"

    if not os.path.exists(model_path):
        print(f"Local model not found at '{model_path}'. Downloading from Roboflow...")
        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("stuff-m0fm7").project("clash-ai-kimrx-instant")
            version = project.version(9)
            version.download("yolov8")
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model from Roboflow: {e}")
            exit()
    else:
        print("Local model found.")
    
    # --- Load Class Names from data.yaml ---
    global NUM_CARD_TYPES, CLASS_NAMES
    try:
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            CLASS_NAMES = data_yaml['names']
            NUM_CARD_TYPES = len(CLASS_NAMES)
        print(f"Loaded {NUM_CARD_TYPES} class names from {yaml_path}.")
    except Exception as e:
        print(f"Error loading class names from {yaml_path}: {e}")
        exit()

    # --- Initialize Local Model, ADB, Bot, and Agent ---
    local_model = YOLO(model_path)
    print("Local YOLOv8 model loaded.")

    try:
        bot = Bot(AdbDevice.list_devices()[0])
    except IndexError:
        print("ERROR: No ADB devices found.")
        exit()
    print(f"Connected to ADB device: {bot.device}.")

    agent = ActorCritic(
        fixed_input_dim=FIXED_INPUT_DIM,
        num_card_types=NUM_CARD_TYPES,
        card_continuous_feature_dim=CARD_CONTINUOUS_DIM,
        num_discrete_actions=NUM_DISCRETE_ACTIONS,
        continuous_action_dim=CONTINUOUS_ACTION_DIM,
        card_embedding_size=CARD_EMBEDDING_SIZE
    )
    agent.eval()
    print("Reinforcement learning agent initialized.")

    print("\nStarting bot...")
    
    try:
        while True:
            # 1. Get state using the local model
            fixed_state, card_ids, card_features = get_state(bot, local_model)

            # 2. Ask agent for action
            with torch.no_grad():
                d_action, c_action, _, _ = agent.select_action(fixed_state, card_ids, card_features)
            
            if d_action < 4:
                print(f"Agent chose to play card {d_action} at position {c_action}")
                scaled_x = (c_action[0] + 1) / 2
                scaled_y = (c_action[1] + 1) / 2
                bot.play_card(d_action, scaled_x, scaled_y)
            else:
                print("Agent chose to do nothing (action 4).")

            sleep(2)

    except KeyboardInterrupt:
        print("\nBot stopped by user.")


if __name__ == "__main__":
    main()