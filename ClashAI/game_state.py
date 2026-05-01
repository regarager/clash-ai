from typing import Any, Optional
from enum import Enum, auto
import torch


class GameScreen(Enum):
    """
    Enum representing the different screens or states the game can be in.
    """

    MAIN_PAGE = auto()
    GAME_SCREEN = auto()  # Active battle
    END_SCREEN = auto()  # Victory/Defeat screen
    UNKNOWN = auto()


class GameState:
    """
    Represents the complete, extracted state of the game from a single screenshot.
    This class holds all game-related data, including elixir, tower healths,
    object detections, and the tensor representations required by the model.
    """

    def __init__(
        self,
        elixir: int,
        tower_healths: Optional[dict[str, float]],
        detections: list[dict[str, Any]],
        fixed_inputs: torch.Tensor,
        card_ids: torch.Tensor,
        card_continuous_features: torch.Tensor,
        playable_mask: torch.Tensor,
        screen_type: GameScreen = GameScreen.UNKNOWN,
    ):
        self.elixir = elixir
        self.tower_healths = tower_healths
        self.detections = detections
        self.fixed_inputs = fixed_inputs
        self.card_ids = card_ids
        self.card_continuous_features = card_continuous_features
        self.playable_mask = playable_mask
        self.screen_type = screen_type

    def __str__(self) -> str:
        """
        Returns a string representation of the game state for easy reading,
        including elixir, tower healths, hand cards, and number of detections.
        """
        output = f"--- Game State ({self.screen_type.name}) ---\n"
        output += f"Elixir: {self.elixir}\n"

        if self.tower_healths:
            output += "Towers:\n"
            for tower, health in sorted(self.tower_healths.items()):
                name = (
                    tower.replace("ally_", "")
                    .replace("enemy_", "")
                    .replace("_", " ")
                    .title()
                )
                output += f"  - {name}: {health:.0%}\n"
        else:
            output += "Towers: Health data not available.\n"

        # Hand Cards (decoded from card_ids)
        from .hand_reader import ALLOWED_TEMPLATES

        whitelist = sorted(list(ALLOWED_TEMPLATES))
        hand_names = []
        for cid in self.card_ids.tolist():
            if cid > 0 and cid <= len(whitelist):
                hand_names.append(whitelist[cid - 1])
            else:
                hand_names.append("unknown")
        output += f"Hand: {', '.join(hand_names)}\n"

        output += f"Detections: {len(self.detections)} objects\n"

        return output
