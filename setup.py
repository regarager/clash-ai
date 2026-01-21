import os
import dotenv
# from roboflow import Roboflow # Removed

def get_api_key():
    """
    Checks for the Roboflow API key in environment variables or a .env file.
    If not found, prompts the user and saves it to a .env file.
    """
    dotenv.load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        print("--- Roboflow API Key Setup ---")
        print("Your Roboflow API key is required to use the Roboflow Inference API.")
        print("You can find your key here: https://universe.roboflow.com/settings/api")
        api_key = input("Please enter your Roboflow API key: ").strip()

        if not api_key:
            print("ERROR: API key cannot be empty.")
            exit()

        with open(".env", "a") as f: # Use "a" to append, not overwrite existing .env content
            f.write(f"\nROBOFLOW_API_KEY={api_key}\n")
        print("Successfully saved API key to .env file.")
        # Load the newly created .env file
        dotenv.load_dotenv()
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    return api_key

def main():
    """
    Main setup script.
    """
    print("--- Clash Royale AI Bot Setup ---")
    get_api_key()
    print("\nSetup complete.")

if __name__ == "__main__":
    main()

