from typing import Any, Optional
import torch

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
    ):
        self.elixir = elixir
        self.tower_healths = tower_healths
        self.detections = detections
        self.fixed_inputs = fixed_inputs
        self.card_ids = card_ids
        self.card_continuous_features = card_continuous_features

    def __str__(self) -> str:
        """
        Returns a string representation of the game state for easy reading,
        including elixir, tower healths, and number of detections.
        """
        output = "--- Game State ---\n"
        output += f"Elixir: {self.elixir}\n"

        if self.tower_healths:
            output += "Towers:\n"
            for tower, health in sorted(self.tower_healths.items()):
                name = tower.replace("ally_", "").replace("enemy_", "").replace("_", " ").title()
                output += f"  - {name}: {health:.0%}\n"
        else:
            output += "Towers: Health data not available.\n"
            
        output += f"Detections: {len(self.detections)} objects\n"

        return output
