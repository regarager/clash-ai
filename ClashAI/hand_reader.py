import cv2
import numpy as np
import os
import time
from typing import List, Dict, Tuple, Set
from .positions import CARDS

class HandReader:
    """
    Identifies the 4 cards currently in the bot's hand.
    Optimized for large card libraries using 16x16 Perceptual Hashing (256 bits).
    """

    def __init__(self, template_dir: str = "setup/card_templates"):
        self.template_dir = template_dir
        self.card_slots = CARDS
        self.crop_w = 110
        self.crop_h = 140
        
        # State for optimization
        self.template_hashes: Dict[str, np.ndarray] = {}
        self.active_deck: Set[str] = set()
        self.max_deck_size = 8
        
        self.templates = self._load_templates()

    def _get_image_hash(self, img: np.ndarray) -> np.ndarray:
        """
        Generates a 256-bit Perceptual Hash (16x16).
        Returns a boolean numpy array for fast comparison.
        """
        # 1. Resize to 16x16
        resized = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
        
        # 2. Grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        # 3. Compute Mean and build hash
        avg = gray.mean()
        return gray > avg

    def _hamming_distance(self, h1: np.ndarray, h2: np.ndarray) -> int:
        """Calculates bit difference between two 16x16 hashes."""
        return np.count_nonzero(h1 != h2)

    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Loads reference card images and pre-computes their hashes."""
        templates = {}
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            return templates

        for filename in os.listdir(self.template_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(self.template_dir, filename)
                img = cv2.imread(path)
                if img is not None:
                    card_name = os.path.splitext(filename)[0]
                    templates[card_name] = img
                    self.template_hashes[card_name] = self._get_image_hash(img)
        
        print(f"HAND_READER: Indexed {len(self.template_hashes)} card hashes (16x16).")
        return templates

    def reset_active_deck(self):
        """Call this at the start of a new match."""
        self.active_deck.clear()

    def _match_card(self, crop: np.ndarray) -> str:
        """Identifies the card using 16x16 hashing."""
        if not self.template_hashes:
            return "unknown"

        crop_hash = self._get_image_hash(crop)
        best_match = "unknown"
        min_dist = 256 # Max distance for 16x16 hash
        
        # 1. Priority Search: Active Deck
        for name in self.active_deck:
            dist = self._hamming_distance(crop_hash, self.template_hashes[name])
            if dist < min_dist:
                min_dist = dist
                best_match = name

        # 10 bits out of 256 is ~96% similarity
        if min_dist <= 10: 
            return best_match

        # 2. Global Search
        if len(self.active_deck) < self.max_deck_size:
            for name, h_val in self.template_hashes.items():
                if name in self.active_deck: continue
                
                dist = self._hamming_distance(crop_hash, h_val)
                if dist < min_dist:
                    min_dist = dist
                    best_match = name

        # Threshold: 50 bits out of 256 is ~80% similarity
        if min_dist > 50:
            return "unknown"
            
        if best_match != "unknown":
            self.active_deck.add(best_match)
            
        return best_match

    def _check_playability(self, crop: np.ndarray) -> bool:
        """Determines if a card is playable based on saturation."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        return avg_saturation > 45 

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
            playable = self._check_playability(crop)
            card_name = self._match_card(crop)
            
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
