import argparse
import os
from warnings import warn

import cv2
import numpy as np
import torch
from ultralytics.models import YOLO


def train_model(device: str):
    if device == "cpu":
        warn("Using CPU to train. Please pass a GPU device if possible.")
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="datasets/data.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
        workers=4,
        device=device,
    )

    return results


def validate_image(image_path: str):
    """Validate that the image file exists and can be read by OpenCV"""
    print(f"Validating image: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")

    # Check file size
    file_size = os.path.getsize(image_path)
    print(f"File size: {file_size} bytes")
    if file_size == 0:
        raise ValueError(f"Image file {image_path} is empty")

    # Check file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    print(f"File extension: {file_ext}")

    # Try to read with OpenCV
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"OpenCV cannot read image file {image_path}")

        print(
            f"Image successfully read: {image_path} (shape: {img.shape}, dtype: {img.dtype})"
        )
        return True

    except Exception as e:
        print(f"Error reading image with OpenCV: {e}")

        # Try alternative reading method
        try:
            with open(image_path, "rb") as f:
                file_bytes = bytearray(f.read())
            img = cv2.imdecode(np.asarray(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Alternative reading method also failed")
            print(
                f"Image successfully read with alternative method (shape: {img.shape})"
            )
            return True
        except Exception as e2:
            raise ValueError(f"All image reading methods failed: {e2}")


def classify_image(model_path: str, image_path: str, confidence_threshold=0.5):
    # Validate image before processing
    validate_image(image_path)

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print("Starting prediction...")
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=True,
        save_txt=True,
    )

    return results


def display_results(results, image_path):
    for r in results:
        print(f"\n--- Detection Results for {image_path} ---")
        print(f"Number of detections: {len(r.boxes)}")

        class_names = r.names

        for i, box in enumerate(r.boxes):
            # Extract coordinates and convert to Python scalars
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().item()  # Use .item() to get Python scalar
            class_id = box.cls[0].cpu().item()  # Use .item() to get Python scalar
            class_id_int = int(class_id)  # Convert to integer
            class_name = class_names[class_id_int]  # Use integer index

            print(f"Detection {i+1}:")
            print(f"  Class: {class_name} (ID: {class_id_int})")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


def _get_devices():
    res = (
        [str(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )

    return res + ["cpu"]


def main():
    parser = argparse.ArgumentParser(description="Clash Royale Image Classification")
    parser.add_argument(
        "--mode",
        choices=["train", "classify"],
        default="classify",
        help="Operation mode",
    )
    parser.add_argument("--image", type=str, help="Path to image for classification")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to model weights for classification",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=_get_devices(),
        help="Device to run training on",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="datasets/data.yaml",
        help="Path to training data YAML file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for classification",
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("Starting model training...")
        train_model(args.device)
        print("Training completed!")

    elif args.mode == "classify":
        if not args.image:
            print("Error: Please provide an image path using --image")
            return

        try:
            print(f"Classifying image: {args.image}")
            results = classify_image(args.model, args.image, args.conf)
            display_results(results, args.image)
        except Exception as e:
            print(f"Error during classification: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check if the image file is not corrupted")
            print("2. Try converting the PNG to JPEG format")
            print("3. Check if the image has valid content")
            print("4. Verify the model file is a valid YOLO model")


if __name__ == "__main__":
    main()
