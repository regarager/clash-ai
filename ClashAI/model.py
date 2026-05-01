import argparse
import os

import cv2
import numpy as np
import torch
from positions import BBox
from ultralytics.models import YOLO

from .logger import *


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
    debug(f"Validating image: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")

    # Check file size
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        raise ValueError(f"Image file {image_path} is empty")

    # Try to read with OpenCV
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"OpenCV cannot read image file {image_path}")

        info(
            f"Image successfully read: {image_path} (shape: {img.shape}, dtype: {img.dtype})"
        )
        return True

    except Exception as e:
        error(f"Error reading image with OpenCV: {e}")

        # Try alternative reading method
        try:
            with open(image_path, "rb") as f:
                file_bytes = bytearray(f.read())
            img = cv2.imdecode(np.asarray(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Alternative reading method also failed")
            info(
                f"Image successfully read with alternative method (shape: {img.shape})"
            )
            return True
        except Exception as e2:
            raise ValueError(f"All image reading methods failed: {e2}")


def classify_image(model_path: str, image_path: str, confidence_threshold=0.5):
    # Validate image before processing
    validate_image(image_path)

    info(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    info("Starting prediction...")
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=True,
        save_txt=True,
    )

    return results


def display_results(results, image_path):
    for r in results:
        info(f"\n--- Detection Results for {image_path} ---")
        info(f"Number of detections: {len(r.boxes)}")

        class_names = r.names

        for i, box in enumerate(r.boxes):
            # Extract coordinates and convert to Python scalars
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = BBox(int(x1), int(y1), int(x2), int(y2))
            confidence = box.conf[0].cpu().item()  # Use .item() to get Python scalar
            class_id = box.cls[0].cpu().item()  # Use .item() to get Python scalar
            class_id_int = int(class_id)  # Convert to integer
            class_name = class_names[class_id_int]  # Use integer index

            info(f"Detection {i + 1}:")
            info(f"  Class: {class_name} (ID: {class_id_int})")
            info(f"  Confidence: {confidence:.4f}")
            info(
                f"  Bounding Box: [{bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f}]"
            )


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
        info("Starting model training...")
        train_model(args.device)
        info("Training completed!")

    elif args.mode == "classify":
        if not args.image:
            error("Error: Please provide an image path using --image")
            return

        try:
            info(f"Classifying image: {args.image}")
            results = classify_image(args.model, args.image, args.conf)
            display_results(results, args.image)
        except Exception as e:
            error(f"Error during classification: {e}")
            error("\nTroubleshooting tips:")
            error("1. Check if the image file is not corrupted")
            error("2. Try converting the PNG to JPEG format")
            error("3. Check if the image has valid content")
            error("4. Verify the model file is a valid YOLO model")


if __name__ == "__main__":
    main()
