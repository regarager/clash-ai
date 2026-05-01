import cv2
import numpy as np
import os
import sys

# Add the project root to sys.path to import ClashAI
sys.path.append(os.getcwd())

from ClashAI.hand_reader import HandReader


def test_perceptual_hashing():
    print("--- Testing Perceptual Hashing Feature ---")

    # Initialize HandReader
    reader = HandReader(template_dir="setup/card_templates")

    # Path to test images (the same ones we just copied as templates)
    test_dir = "setup/card_templates"
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".png")]

    if not test_files:
        print("ERROR: No test files found in setup/card_templates.")
        return

    print(f"Found {len(test_files)} test files.")

    success_count = 0
    for filename in test_files:
        path = os.path.join(test_dir, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"FAILED to read {path}")
            continue

        # The filename (without extension) is the expected card name
        expected_name = os.path.splitext(filename)[0]

        # Test _match_card directly
        detected_name = reader._match_card(img)

        # Calculate distance for debugging
        if detected_name != "unknown":
            h1 = reader._get_image_hash(img)
            h2 = reader.template_hashes[detected_name]
            dist = reader._hamming_distance(h1, h2)
            print(
                f"Test: {expected_name} -> Detected: {detected_name} (Hamming Distance: {dist})"
            )
        else:
            print(f"Test: {expected_name} -> Detected: {detected_name}")

        if detected_name == expected_name:
            success_count += 1
        else:
            print(f"  MISMATCH: Expected {expected_name}, got {detected_name}")

    print(f"\nResults: {success_count}/{len(test_files)} correct matches.")

    if success_count == len(test_files):
        print("SUCCESS: Perceptual hashing identified all templates correctly.")
    else:
        print("FAILURE: Some templates were not identified correctly.")


if __name__ == "__main__":
    test_perceptual_hashing()
