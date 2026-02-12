import unittest
import numpy as np
import cv2
import os

from vision import get_elixir, get_tower_healths, is_game_active, get_full_game_state
from positions import ELIXIR_BAR_BBOX, BBox
from game_state import GameState # Import GameState

class TestVisionFunctions(unittest.TestCase):
    # Declare class attributes with type hints for mypy
    blank_image: np.ndarray
    image_path: str
    tower_bboxes: dict[str, BBox]


    @classmethod
    def setUpClass(cls):
        # Define bboxes using the default constructor
        cls.tower_bboxes = {
            "ally_king_tower":          BBox(480, 1450, 600, 1460),
            "ally_left_princess_tower": BBox(420, 1070, 530, 1090),
            "ally_right_princess_tower":BBox(930, 1070, 1040, 1090),
            "enemy_king_tower":         BBox(480, 350, 600, 360),
            "enemy_left_princess_tower":BBox(420, 190, 530, 210),
            "enemy_right_princess_tower":BBox(930, 190, 1040, 210),
        }
        all_bboxes = list(cls.tower_bboxes.values()) + [ELIXIR_BAR_BBOX]
        
        max_x = max(b.x2 for b in all_bboxes)
        max_y = max(b.y2 for b in all_bboxes)
        
        cls.blank_image = np.zeros((max_y + 10, max_x + 10, 3), dtype=np.uint8)
        cls.image_path = "temp_test_image.png"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.image_path):
            os.remove(cls.image_path)

    def _create_mock_image(self, bboxes_to_fill: list[BBox], fill_map: dict):
        mock_image = self.blank_image.copy()
        for bbox in bboxes_to_fill:
            x, y, w, h = bbox.to_xywh()
            x, y, w, h = max(0, x), max(0, y), min(w, mock_image.shape[1] - x), min(h, mock_image.shape[0] - y)

            if w <= 0 or h <= 0:
                continue

            # Default to full-width black fill
            fill_color, fill_percentage, empty_color = fill_map.get(bbox, ((0,0,0), 1.0, (0,0,0)))
            
            filled_width = int(w * fill_percentage)
            if filled_width > 0:
                mock_image[y:y+h, x:x+filled_width] = fill_color
            if filled_width < w:
                mock_image[y:y+h, x+filled_width:x+w] = empty_color
        
        cv2.imwrite(self.image_path, mock_image)
        return self.image_path

    def test_get_tower_healths_empty(self):
        empty_ally_color = (79, 56, 45)
        empty_enemy_color = (51, 35, 65)
        
        fill_map = {
            self.tower_bboxes["ally_king_tower"]: (empty_ally_color, 1.0, empty_ally_color),
            self.tower_bboxes["ally_left_princess_tower"]: (empty_ally_color, 1.0, empty_ally_color),
            self.tower_bboxes["ally_right_princess_tower"]: (empty_ally_color, 1.0, empty_ally_color),
            self.tower_bboxes["enemy_king_tower"]: (empty_enemy_color, 1.0, empty_enemy_color),
            self.tower_bboxes["enemy_left_princess_tower"]: (empty_enemy_color, 1.0, empty_enemy_color),
            self.tower_bboxes["enemy_right_princess_tower"]: (empty_enemy_color, 1.0, empty_enemy_color),
        }

        self._create_mock_image(list(self.tower_bboxes.values()), fill_map)
        healths = get_tower_healths(self.image_path)
        for tower, health in healths.items():
            self.assertAlmostEqual(health, 0.0, places=2)

    def test_get_tower_healths_full(self):
        ally_health_color = (178, 147, 80)
        enemy_health_color = (110, 83, 176)
        
        fill_map = {
            self.tower_bboxes["ally_king_tower"]: (ally_health_color, 1.0, (0,0,0)),
            self.tower_bboxes["ally_left_princess_tower"]: (ally_health_color, 1.0, (0,0,0)),
            self.tower_bboxes["ally_right_princess_tower"]: (ally_health_color, 1.0, (0,0,0)),
            self.tower_bboxes["enemy_king_tower"]: (enemy_health_color, 1.0, (0,0,0)),
            self.tower_bboxes["enemy_left_princess_tower"]: (enemy_health_color, 1.0, (0,0,0)),
            self.tower_bboxes["enemy_right_princess_tower"]: (enemy_health_color, 1.0, (0,0,0)),
        }

        self._create_mock_image(list(self.tower_bboxes.values()), fill_map)
        healths = get_tower_healths(self.image_path)
        for tower, health in healths.items():
            self.assertAlmostEqual(health, 1.0, places=2)

    def test_get_tower_healths_partial(self):
        empty_ally_color = (79, 56, 45)
        ally_health_color = (178, 147, 80)
        bbox = self.tower_bboxes["ally_king_tower"]
        
        fill_map = {bbox: (ally_health_color, 0.5, empty_ally_color)}
        self._create_mock_image([bbox], fill_map)
        healths = get_tower_healths(self.image_path)
        self.assertAlmostEqual(healths["ally_king_tower"], 0.5, places=2)

    def test_get_elixir(self):
        purple_color = (255, 0, 255)
        fill_map = {ELIXIR_BAR_BBOX: (purple_color, 1.0, (0,0,0))}
        self._create_mock_image([ELIXIR_BAR_BBOX], fill_map)
        self.assertEqual(get_elixir(self.image_path), 10)

        fill_map = {ELIXIR_BAR_BBOX: ((0,0,0), 0.0, (0,0,0))}
        self._create_mock_image([ELIXIR_BAR_BBOX], fill_map)
        self.assertEqual(get_elixir(self.image_path), 0)

    def test_is_game_active(self):
        # Test active when purple elixir is present
        purple_color = (255, 0, 255)
        fill_map = {ELIXIR_BAR_BBOX: (purple_color, 1.0, (0,0,0))}
        self._create_mock_image([ELIXIR_BAR_BBOX], fill_map)
        self.assertTrue(is_game_active(self.image_path))

        # Test active when empty elixir color is present
        empty_elixir_color = (130, 80, 0) # BGR for an empty bar section
        fill_map = {ELIXIR_BAR_BBOX: (empty_elixir_color, 1.0, empty_elixir_color)}
        self._create_mock_image([ELIXIR_BAR_BBOX], fill_map)
        self.assertTrue(is_game_active(self.image_path))

        # Test not active when neither color is present
        fill_map = {ELIXIR_BAR_BBOX: ((0,0,0), 1.0, (0,0,0))} # Completely black bar
        self._create_mock_image([ELIXIR_BAR_BBOX], fill_map)
        self.assertFalse(is_game_active(self.image_path))

    def test_get_game_state_inactive(self):
        # Test that get_game_state returns None when the game is not active
        # The bot needs to be mocked to create a full GameState object for the function to work correctly
        class MockBot:
            image_path: str # Added for mypy
            def screenshot(self) -> str: return self.image_path
            def get_screen_size(self) -> tuple[int, int]: return (1080, 1920)

        mock_bot = MockBot()

        fill_map = {ELIXIR_BAR_BBOX: ((0,0,0), 1.0, (0,0,0))} # Completely black bar
        self._create_mock_image([ELIXIR_BAR_BBOX], fill_map)
        
        # Now get_full_game_state returns a GameState object, not None
        game_state_obj = get_full_game_state(mock_bot)
        
        # Check if elixir is 0, which would indicate an inactive state (based on how get_elixir works)
        self.assertEqual(game_state_obj.elixir, 0)
        self.assertIsInstance(game_state_obj, GameState)


if __name__ == "__main__":
    unittest.main()
