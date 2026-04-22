import cv2
import numpy as np
import os
import time
from typing import List, Dict, Tuple, Set
from .positions import CARDS

# Hardcoded whitelist of cards we have templates for.
# This prevents the bot from ever identifying a card not in our library.
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
    Uses center-weighted dHash for robust recognition.
    """

    def __init__(self, template_dir: str = "setup/card_templates"):
        self.template_dir = template_dir
        self.card_slots = CARDS
        self.crop_w = 110
        self.crop_h = 140
        
        # State for optimization
        self.template_hashes: Dict[str, np.ndarray] = {}
        self.active_deck: Set[str] = set()
        
        self.templates = self._load_templates()

    def _get_image_hash(self, img: np.ndarray) -> np.ndarray:
        """
        Generates a 256-bit Difference Hash (dHash) from the center of the image.
        Focuses on the character art and ignores noisy edges and backgrounds.
        """
        if img is None or img.size == 0:
            return np.zeros(256, dtype=bool)

        # 1. Center Crop (focus on the central 60% of the card)
        h, w = img.shape[:2]
        ch, cw = int(h * 0.6), int(w * 0.6)
        y1, x1 = (h - ch) // 2, (w - cw) // 2
        center_img = img[y1:y1+ch, x1:x1+cw]

        # 2. Resize to 17x16 (for 16x16 differences)
        resized = cv2.resize(center_img, (17, 16), interpolation=cv2.INTER_AREA)
        
        # 3. Grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        # 4. Compute differences between horizontal pixels
        diff = gray[:, 1:] > gray[:, :-1]
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
                
                # Only load if it's in our allowed whitelist
                if card_name not in ALLOWED_TEMPLATES:
                    continue

                path = os.path.join(self.template_dir, filename)
                img = cv2.imread(path)
                if img is not None:
                    templates[card_name] = img
                    self.template_hashes[card_name] = self._get_image_hash(img)
        
        print(f"HAND_READER: Indexed {len(self.template_hashes)} card hashes (Precision-dHash 16x16).")
        return templates

    def reset_active_deck(self):
        """Call this at the start of a new match."""
        self.active_deck.clear()

    def _match_card(self, crop: np.ndarray) -> str:
        """Identifies the card using center-weighted 16x16 dHash."""
        if not self.template_hashes:
            return "unknown"

        # 1. Reject solid or very low-contrast crops (empty slots)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray_crop)
        if std_dev < 10: 
            return "unknown"

        crop_hash = self._get_image_hash(crop)
        best_match = "unknown"
        min_dist = 256 
        
        # 2. Global Search (Identical to Unit Test logic)
        for name, h_val in self.template_hashes.items():
            dist = self._hamming_distance(crop_hash, h_val)
            if dist < min_dist:
                min_dist = dist
                best_match = name

        # Strict Precision threshold: 50 bits out of 256 (~80% similarity)
        # Only return a match if it's high-confidence and in our whitelist.
        if min_dist > 50:
            return "unknown"
            
        return best_match

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
        for i, pos in enumerate(self.card_slots):
            crop = self._get_crop(image, pos)
            
            # Identify the card name (Always matches against whitelist)
            card_name = self._match_card(crop)
            
            # Check playability for the agent's action mask
            playable = self._check_playability(crop)
            
            hand.append({
                "slot": i,
                "name": card_name,
                "playable": playable
            })
        return hand

    def save_hand_crops(self, image: np.ndarray, output_dir: str = "screenshots/hand_crops"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = int(time.time())
        for i, pos in enumerate(self.card_slots):
            crop = self._get_crop(image, pos)
            cv2.imwrite(os.path.join(output_dir, f"hand_{i}_{timestamp}.png"), crop)
