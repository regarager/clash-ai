import cv2
import numpy as np
import os
import time
from typing import List, Dict, Tuple, Set
from .positions import CARDS

class HandReader:
    """
    Identifies the 4 cards currently in the bot's hand.
    Optimized for large card libraries using Perceptual Hashing and Active Deck filtering.
    """

    def __init__(self, template_dir: str = "setup/card_templates"):
        self.template_dir = template_dir
        self.card_slots = CARDS
        self.crop_w = 110
        self.crop_h = 140
        
        # State for optimization
        self.template_hashes: Dict[str, int] = {}
        self.active_deck: Set[str] = set()
        self.max_deck_size = 8
        
        self.templates = self._load_templates()

    def _get_image_hash(self, img: np.ndarray) -> int:
        """
        Generates a 64-bit Perceptual Hash (Average Hash).
        Resizes to 8x8, grayscales, and computes bits based on mean brightness.
        """
        # 1. Resize to 8x8 (removes high-frequency detail)
        resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        
        # 2. Grayscale (removes color variance)
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        # 3. Compute Mean
        avg = gray.mean()
        
        # 4. Build 64-bit hash (1 if pixel > mean, else 0)
        # We use a bitstring converted to an integer
        diff = gray > avg
        hash_val = 0
        for i, v in enumerate(diff.flatten()):
            if v:
                hash_val |= (1 << i)
        return hash_val

    def _hamming_distance(self, h1: int, h2: int) -> int:
        """Calculates how many bits differ between two hashes."""
        return bin(h1 ^ h2).count('1')

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
                    # Pre-compute hash for fast matching later
                    self.template_hashes[card_name] = self._get_image_hash(img)
        
        print(f"HAND_READER: Indexed {len(self.template_hashes)} card hashes.")
        return templates

    def reset_active_deck(self):
        """Call this at the start of a new match."""
        self.active_deck.clear()

    def _match_card(self, crop: np.ndarray) -> str:
        """Identifies the card using fast hashing and active deck priority."""
        if not self.template_hashes:
            return "unknown"

        crop_hash = self._get_image_hash(crop)
        best_match = "unknown"
        min_dist = 64 # Max distance for 64-bit hash
        
        # 1. Priority Search: Check cards already seen in this deck
        for name in self.active_deck:
            dist = self._hamming_distance(crop_hash, self.template_hashes[name])
            if dist < min_dist:
                min_dist = dist
                best_match = name

        # If we found a very strong match in the active deck, return early (Optimization)
        if min_dist <= 5: 
            return best_match

        # 2. Global Search: Only if we haven't found a perfect match in the active deck
        # and we haven't filled the deck yet.
        if len(self.active_deck) < self.max_deck_size:
            for name, h_val in self.template_hashes.items():
                if name in self.active_deck: continue # Already checked
                
                dist = self._hamming_distance(crop_hash, h_val)
                if dist < min_dist:
                    min_dist = dist
                    best_match = name

        # Threshold: 12 bits out of 64 is ~80% similarity
        if min_dist > 12:
            return "unknown"
            
        # Add to active deck if identified
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
