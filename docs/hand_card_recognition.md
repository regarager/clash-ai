# Hand Card Recognition Module: Implementation Outline

This document outlines the strategy for implementing a module that identifies the 4 cards currently in the bot's hand.

## 1. Core Challenges
*   **Overlay Interference:** In-game hand cards often have elixir cost bubbles, level markers, or "evolution" glows that aren't in the raw card art.
*   **Playability State:** Cards are desaturated (greyed out) when the player has insufficient elixir.
*   **Performance:** Recognition must be fast enough to run every few frames without lagging the agent.

## 2. Proposed Architecture: `HandReader`

### A. Data Preparation
1.  **Template Library:** Create a folder of "Reference Hand Crops." Since hand cards look slightly different than collection cards, we should save 100x100 pixel crops of each card *as it appears in the hand* during a match.
2.  **Masking:** Define a binary mask to ignore the corners of the card where the Elixir cost (bottom-left) and Level (top-left/right) typically reside. This focuses the comparison on the central character art.

### B. The Recognition Pipeline
For each of the 4 hand positions defined in `positions.py`:

1.  **Extract ROI:** Crop the 110x140 area corresponding to the card slot.
2.  **Determine Playability:**
    *   Calculate the average saturation (S channel in HSV).
    *   If saturation is below a threshold (e.g., < 20%), the card is "Greyed Out" (Inavailable).
    *   *Action:* If greyed out, we can still identify it, but we flag it as `unplayable` in the `GameState`.
3.  **Preprocessing:**
    *   Convert to Grayscale (to make recognition work whether the card is greyed out or colored).
    *   Apply a Gaussian Blur to reduce noise from the overlay.
4.  **Identification (Matching):**
    *   **Option 1: Template Matching:** Use `cv2.matchTemplate` with `TM_CCOEFF_NORMED` against the template library.
    *   **Option 2: Perceptual Hashing (Recommended):** Use `imagehash` (dHash or phash). It is extremely fast and robust to the "minor variations" caused by overlays. Compare the hash of the crop to a pre-calculated dictionary of card hashes.

### C. Module Structure
```python
class HandReader:
    def __init__(self, template_dir: str):
        # Load hashes/templates for all 100+ cards
        self.reference_hashes = self._load_references(template_dir)
        # Precise centers for 1440x2560 resolution
        self.card_slots = [(528, 1544), (708, 1544), (892, 1544), (1076, 1544)]

    def identify_hand(self, screenshot) -> list[dict]:
        hand = []
        for i, pos in enumerate(self.card_slots):
            crop = self._get_crop(screenshot, pos)
            is_playable = self._check_saturation(crop)
            card_name = self._match_card(crop)
            hand.append({
                "slot": i,
                "name": card_name,
                "playable": is_playable
            })
        return hand
```

## 3. Integration with `ActorCritic`
*   **GameState Update:** The `hand_card_ids` (index 0-3) should be passed into the `ActorCritic` model.
*   **Input Embedding:** The model will now "know" exactly which card is in Slot 0, allowing it to learn that "Slot 0 (Giant) should be placed at the back" vs "Slot 0 (Arrows) should be placed on enemies."

## 4. Next Steps
1.  **Collect Samples:** Run the bot and save crops of the hand area to build the reference library.
2.  **Implement `HandReader`:** Create the class in `ClashAI/hand_reader.py`.
3.  **Update `vision.py`:** Call `HandReader` inside `get_full_game_state` and populate the `card_ids` tensor specifically with the hand cards.
