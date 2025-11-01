import argparse
import os
from warnings import warn

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


def classify_image(model_path: str, image_path: str, confidence_threshold=0.5):
    model = YOLO(model_path)

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
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = box.cls[0].cpu().numpy().astype(int)
            class_name = class_names[class_id]

            print(f"Detection {i+1}:")
            print(f"  Class: {class_name} (ID: {class_id})")
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
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to model weights for classification",
    )
    parser.add_argument("--image", type=str, help="Path to image for classification")
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

        if not os.path.exists(args.image):
            print(f"Error: Image file {args.image} does not exist")
            return

        if not os.path.exists(args.model):
            print(f"Error: Model file {args.model} does not exist")
            print("Please train the model first or provide correct model path")
            return

        print(f"Classifying image: {args.image}")
        results = classify_image(args.model, args.image, args.conf)
        display_results(results, args.image)


if __name__ == "__main__":
    main()
