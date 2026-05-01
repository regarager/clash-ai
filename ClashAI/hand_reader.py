import cv2
import numpy as np
import os
import time
from typing import List, Dict, Tuple, Set
from .positions import CARDS

# Hardcoded whitelist of cards we have templates for.
ALLOWED_TEMPLATES = {
    "battle-ram",
    "bomber",
    "giant",
    "knight",
    "mini-pekka",
    "minions",
    "musketeer",
    "valkyrie",
}


class HandReader:
    """
    Identifies the 4 cards currently in the bot's hand.
    Uses high-precision jitter-robust translation-invariant dHash.
    """

    def __init__(self, template_dir: str = "setup/card_templates"):
        self.template_dir = template_dir
        self.card_slots = CARDS
        self.crop_w = 110
        self.crop_h = 140

        # State for optimization and persistence
        self.template_hashes: Dict[str, np.ndarray] = {}
        self.active_deck: Set[str] = set()
        self.last_hand_names: List[str] = ["unknown"] * 4

        self.templates = self._load_templates()

    def _get_image_hash(self, img: np.ndarray) -> np.ndarray:
        """
        Generates a 4096-bit translation-invariant hash.
        Uses a 64x64 grid and Otsu thresholding for extreme precision.
        """
        if img is None or img.size == 0:
            return np.zeros(4096, dtype=bool)

        # 1. Pre-process
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Find "Content" Bounding Box (Otsu Thresholding)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Use Otsu to find character edges
        _, thresh = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            content_img = gray[
                max(0, y1) : min(gray.shape[0], y2), max(0, x1) : min(gray.shape[1], x2)
            ]
        else:
            h, w = gray.shape[:2]
            ch, cw = int(h * 0.7), int(w * 0.7)
            y1, x1 = (h - ch) // 2, (w - cw) // 2
            content_img = gray[y1 : y1 + ch, x1 : x1 + cw]

        # 3. Resize and dHash (65x64 for 64x64 bits = 4096 bits)
        resized = cv2.resize(content_img, (65, 64), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.flatten()

    def _hamming_distance(self, h1: np.ndarray, h2: np.ndarray) -> int:
        """Calculates bit difference between two hashes."""
        return np.count_nonzero(h1 != h2)

    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Loads reference card images and pre-computes their hashes."""
        templates = {}
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            return templates

        for filename in os.listdir(self.template_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                card_name = os.path.splitext(filename)[0]
                if card_name not in ALLOWED_TEMPLATES:
                    continue

                path = os.path.join(self.template_dir, filename)
                img = cv2.imread(path)
                if img is not None:
                    templates[card_name] = img
                    self.template_hashes[card_name] = self._get_image_hash(img)

        print(
            f"HAND_READER: Indexed {len(self.template_hashes)} card hashes (Robust-dHash 64x64)."
        )
        return templates

    def reset_active_deck(self):
        """Call this at the start of a new match."""
        self.active_deck.clear()
        self.last_hand_names = ["unknown"] * 4

    def _match_card(self, image: np.ndarray, slot_idx: int) -> str:
        """
        Identifies the card using Jitter-Robust matching.
        """
        if not self.template_hashes:
            return "unknown"

        pos = self.card_slots[slot_idx]
        offsets = [(0, 0), (0, -4), (0, -8), (-2, 0), (2, 0)]

        best_overall_match = "unknown"
        min_overall_dist = 4096
        best_crop_hash = None

        for dx, dy in offsets:
            jitter_pos = (pos[0] + dx, pos[1] + dy)
            crop = self._get_crop(image, jitter_pos)

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if np.std(gray_crop) < 10:
                continue

            current_hash = self._get_image_hash(crop)

            for name, t_hash in self.template_hashes.items():
                dist = self._hamming_distance(current_hash, t_hash)
                if dist < min_overall_dist:
                    min_overall_dist = dist
                    best_overall_match = name
                    best_crop_hash = current_hash

        if best_crop_hash is None:
            return "unknown"

        # Apply thresholds (1500 for 4096-bit hash, approx 36% error budget)
        detected_name = best_overall_match if min_overall_dist <= 1500 else "unknown"

        if detected_name == "unknown":
            prev_name = self.last_hand_names[slot_idx]
            if prev_name != "unknown":
                prev_hash = self.template_hashes.get(prev_name)
                sticky_dist = self._hamming_distance(best_crop_hash, prev_hash)
                if sticky_dist < 1800:
                    detected_name = prev_name

        if detected_name != "unknown":
            self.last_hand_names[slot_idx] = detected_name

        return detected_name

    def _check_playability(self, crop: np.ndarray) -> bool:
        """Determines if a card is playable based on saturation."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        return avg_saturation > 30

    def _get_crop(self, image: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
        x_center, y_center = pos
        x1 = max(0, x_center - self.crop_w // 2)
        y1 = max(0, y_center - self.crop_h // 2)
        x2 = min(image.shape[1], x_center + self.crop_w // 2)
        y2 = min(image.shape[0], y_center + self.crop_h // 2)
        return image[y1:y2, x1:x2]

    def identify_hand(self, image: np.ndarray) -> List[Dict]:
        """Analyzes the image and returns the status of the 4 hand cards."""
        hand = []
        for i in range(len(self.card_slots)):
            card_name = self._match_card(image, i)

            crop = self._get_crop(image, self.card_slots[i])
            playable = self._check_playability(crop)

            hand.append({"slot": i, "name": card_name, "playable": playable})

        return hand

    def save_hand_crops(
        self, image: np.ndarray, output_dir: str = "screenshots/hand_crops"
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = int(time.time())
        for i, pos in enumerate(self.card_slots):
            crop = self._get_crop(image, pos)
            cv2.imwrite(os.path.join(output_dir, f"hand_{i}_{timestamp}.png"), crop)
