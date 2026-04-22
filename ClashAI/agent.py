from time import sleep
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing_extensions import override

from .bot import Bot
from .game_state import GameScreen, GameState
from .vision import calculate_reward, get_full_game_state, reset_hand_reader


__all__ = ["ActorCritic"]




class ActorCritic(nn.Module):
    """
    An Actor-Critic model for a hybrid action space (discrete + continuous)
    and mixed input types, updated to properly handle categorical card IDs.
    """

    fixed_mlp: nn.Sequential
    card_embedding: nn.Embedding
    attention: nn.MultiheadAttention
    cards_mlp: nn.Sequential
    discrete_action_head: nn.Sequential
    continuous_action_head: nn.Sequential
    continuous_action_log_std: nn.Parameter
    critic_head: nn.Sequential
    optimizer: torch.optim.Adam
    rewards: list[float]
    log_probs: list[torch.Tensor]
    state_values: list[torch.Tensor]

    def __init__(
        self,
        bot: Bot,
        fixed_input_dim: int,
        num_card_types: int,
        card_continuous_feature_dim: int,
        num_discrete_actions: int,
        continuous_action_dim: int,
        card_embedding_size: int = 16,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
    ):
        super(ActorCritic, self).__init__()
        self.bot = bot
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.steps_since_update = 0
        self.num_card_slots = num_discrete_actions - 1

        self.fixed_mlp = nn.Sequential(
            nn.Linear(fixed_input_dim, hidden_dim // 2), nn.ReLU()
        )
        self.card_embedding = nn.Embedding(num_card_types + 1, card_embedding_size)
        
        # The attention mechanism now operates only on the card embeddings (size 16)
        attn_dim = card_embedding_size 
        nhead = (
            4 if attn_dim % 4 == 0 else (2 if attn_dim % 2 == 0 else 1)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=nhead, batch_first=True
        )
        self.cards_mlp = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim // 2), nn.ReLU()
        )

        combined_feature_dim = hidden_dim

        self.discrete_action_head = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_discrete_actions),
        )
        self.continuous_action_head = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_card_slots * continuous_action_dim),
        )
        self.continuous_action_log_std = nn.Parameter(
            torch.zeros(self.num_card_slots, continuous_action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def step(
        self,
        previous_state: Optional[GameState],
    ) -> GameState:
        """
        Performs one step of the game loop: gets state, selects action,
        calculates reward, and updates the agent.
        """
        print("\n--- Agent Step Initiated ---")

        # 1. Get current game state
        current_state = get_full_game_state(self.bot)
        print(f"Current Game State: {current_state}")

        # --- Handle UI Screens (Non-Battle) ---
        if current_state.screen_type == GameScreen.END_SCREEN:
            print("BOT: End screen detected. Clicking return to main menu.")
            self.bot.tap((800, 1540))
            sleep(2)
            return current_state

        if current_state.screen_type == GameScreen.MAIN_PAGE:
            print("BOT: Main page detected. Clicking Battle and resetting HandReader.")
            reset_hand_reader()
            from .positions import BATTLE
            self.bot.tap(BATTLE)
            sleep(2)
            return current_state

        # --- Handle Active Battle Screen (Agent Actions) ---
        if current_state.screen_type != GameScreen.GAME_SCREEN:
            print(f"BOT: Non-game screen ({current_state.screen_type.name}) - Clicking (700, 800) to skip.")
            self.bot.tap((700, 800))
            return current_state

        if not current_state.detections:
            print(
                "AGENT DEBUG: Skipping step due to zero detections from vision module."
            )
            return current_state  # Skip the rest of the step

        if previous_state is None:
            return current_state

        # 2. Ask agent for action
        d_action, c_action = self.select_action(
            current_state.fixed_inputs,
            current_state.card_ids,
            current_state.card_continuous_features,
            current_state.playable_mask,
        )
        print(
            f"Agent selected Discrete Action: {d_action}, Continuous Action: {c_action}"
        )

        # 3. Take action
        if d_action < self.num_card_slots:  # Assuming actions 0-3 are "play card"
            scaled_x = (c_action[0] + 1) / 2
            scaled_y = (c_action[1] + 1) / 2
            print(
                f"Playing card {d_action} at scaled position ({scaled_x:.2f}, {scaled_y:.2f})"
            )
            self.bot.play_card(d_action, scaled_x, scaled_y)
        else:
            print(f"Agent chose to do nothing (action {d_action}).")

        sleep(1)  # Wait for the game to update after an action
        print("AGENT: Finished waiting.")

        # 4. Calculate reward based on state change
        print("AGENT: Calculating reward...")
        reward = calculate_reward(current_state, previous_state)
        self.rewards.append(reward)
        print(f"AGENT: Reward calculated: {reward}")

        self.steps_since_update += 1

        # 5. Update the agent periodically or on game end
        # We update if we have enough steps or if a big reward (win/loss) is detected
        if self.steps_since_update >= 10 or abs(reward) >= 500:
            print(f"AGENT: Updating agent after {self.steps_since_update} steps...")
            self.update()
            self.steps_since_update = 0
            print("AGENT: Agent update complete.")
        else:
            print(f"AGENT: Buffering reward ({self.steps_since_update}/10 steps to update).")

        return current_state

    @override
    def forward(
        self,
        fixed_inputs: torch.Tensor,
        card_ids: torch.Tensor | None,
        card_continuous_features: torch.Tensor | None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[Categorical, Normal, torch.Tensor]:
        fixed_features = self.fixed_mlp(fixed_inputs)

        # 1. Process Hand Card Embeddings (card_ids should be size 4)
        if card_ids is not None and card_ids.nelement() > 0:
            hand_embeds = self.card_embedding(card_ids) # [batch, 4, embed_dim]
            
            # 2. Process Field Unit Features (card_continuous_features can vary)
            if (
                card_continuous_features is not None
                and card_continuous_features.nelement() > 0
            ):
                field_features = card_continuous_features # [batch, num_units, 4]
                pass

            attn_output, _ = self.attention(
                hand_embeds, hand_embeds, hand_embeds
            )
            card_features_agg = attn_output.mean(dim=1)
            card_features = self.cards_mlp(card_features_agg)
        else:
            card_features = torch.zeros_like(fixed_features)

        combined_features = torch.cat([fixed_features, card_features], dim=1)

        discrete_logits = self.discrete_action_head(combined_features)
        
        # Apply action mask if provided
        if action_mask is not None:
            # action_mask is [batch, 4] where True means playable.
            # We need to map it to [batch, 5] (4 cards + 1 "do nothing")
            # "Do nothing" (last index) is always allowed.
            batch_size = discrete_logits.shape[0]
            extended_mask = torch.cat(
                [action_mask, torch.ones((batch_size, 1), device=action_mask.device, dtype=torch.bool)],
                dim=1
            )
            # Apply large negative value to unplayable actions
            discrete_logits[~extended_mask] = -1e10

        discrete_dist = Categorical(logits=discrete_logits)

        # Parameterized continuous action: mean for each card slot
        continuous_params = self.continuous_action_head(combined_features)
        continuous_mean = torch.tanh(continuous_params.view(-1, self.num_card_slots, 2))
        
        # Action variance (shared across batch, but separate for each card slot)
        action_std = self.continuous_action_log_std.exp().unsqueeze(0).expand_as(continuous_mean)
        continuous_dist = Normal(continuous_mean, action_std)

        state_value = self.critic_head(combined_features)
        return discrete_dist, continuous_dist, state_value

    def select_action(
        self,
        fixed_inputs: torch.Tensor,
        card_ids: torch.Tensor | None,
        card_continuous_features: torch.Tensor | None,
        playable_mask: torch.Tensor | None = None,
    ) -> tuple[int, np.ndarray[Any, Any]]:
        if fixed_inputs.dim() == 1:
            fixed_inputs = fixed_inputs.unsqueeze(0)
        if card_ids is not None and card_ids.nelement() > 0:
            if card_ids.dim() == 1:
                card_ids = card_ids.unsqueeze(0)
            if (
                card_continuous_features is not None
                and card_continuous_features.dim() == 2
            ):
                card_continuous_features = card_continuous_features.unsqueeze(0)
        if playable_mask is not None and playable_mask.dim() == 1:
            playable_mask = playable_mask.unsqueeze(0)

        discrete_dist, continuous_dist, state_value = self.forward(
            fixed_inputs, card_ids, card_continuous_features, playable_mask
        )

        discrete_action = discrete_dist.sample()
        discrete_idx = int(discrete_action.item())
        
        discrete_log_prob = discrete_dist.log_prob(discrete_action)

        if discrete_idx < self.num_card_slots:
            # Sample position for the selected card from its specific distribution
            card_mean = continuous_dist.loc[:, discrete_idx, :]
            card_std = continuous_dist.scale[:, discrete_idx, :]
            card_dist = Normal(card_mean, card_std)
            
            continuous_action_sampled = card_dist.sample()
            continuous_log_prob = card_dist.log_prob(continuous_action_sampled).sum(dim=-1)
            
            continuous_action_np = continuous_action_sampled.squeeze(0).detach().cpu().numpy()
        else:
            # "Do nothing" action has no associated continuous parameter log-prob
            continuous_log_prob = torch.tensor([0.0], device=discrete_action.device)
            continuous_action_np = np.zeros(2)

        self.log_probs.append(discrete_log_prob + continuous_log_prob)
        self.state_values.append(state_value)

        return discrete_idx, continuous_action_np

    def update(self, gamma: float = 0.99) -> None:
        if not self.rewards:
            return

        rewards = torch.tensor(self.rewards, dtype=torch.float32)

        # If all rewards are the same (e.g., all zeros), stddev will be 0.
        # This can lead to NaNs in normalized returns and subsequent loss calculation.
        # Skip update if rewards are essentially constant, as there's no learning signal.
        if (
            len(rewards) < 2 or rewards.std() < 1e-6
        ):  # Check if std is too small or only one reward
            print(
                f"AGENT WARNING: Skipping update due to constant or insufficient rewards. Stddev: {rewards.std().item()}"
            )
            self.rewards.clear()
            self.log_probs.clear()
            self.state_values.clear()
            return

        log_probs = torch.stack(self.log_probs)
        state_values = torch.stack(self.state_values).squeeze()

        returns: list[float] = []
        discounted_reward: float = 0.0
        for r in reversed(rewards):
            discounted_reward = r + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns_tensor = torch.tensor(returns)

        # Normalize returns
        # Add a small epsilon to the std to prevent division by zero, even if std is already 1e-9
        returns_std = returns_tensor.std()
        if (
            returns_std == 0
        ):  # Ensure we don't divide by zero if all returns are identical
            returns_tensor = returns_tensor - returns_tensor.mean()  # Just center it
        else:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_std + 1e-9
            )

        advantage = returns_tensor - state_values.detach()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(state_values, returns_tensor)
        loss = actor_loss + 0.5 * critic_loss

        # Check for NaNs in loss before backpropagation
        if torch.isnan(loss):
            print(
                "AGENT WARNING: NaN detected in loss. Skipping backpropagation and resetting buffers."
            )
            self.rewards.clear()
            self.log_probs.clear()
            self.state_values.clear()
            return

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards.clear()
        self.log_probs.clear()
        self.state_values.clear()

    def save_model(self, path: str = "clash_ai_agent.pth") -> None:
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = "clash_ai_agent.pth") -> None:
        try:
            self.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"No model found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_value(
        self,
        fixed_inputs: torch.Tensor,
        card_ids: torch.Tensor | None,
        card_continuous_features: torch.Tensor | None,
        playable_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, state_value = self.forward(
            fixed_inputs, card_ids, card_continuous_features, playable_mask
        )
        return state_value


if __name__ == "__main__":
    # Example Usage
    FIXED_INPUT_DIM = 9
    NUM_CARD_TYPES = 110
    CARD_CONTINUOUS_DIM = 4
    NUM_DISCRETE_ACTIONS = 5
    CONTINUOUS_ACTION_DIM = 2
    CARD_EMBEDDING_SIZE = 16

    # You would need a mock Bot object to run this example
    class MockBot(Bot):
        def __init__(self):
            super().__init__(use_adb=False)

        def screenshot(self) -> str:
            return "mock_screenshot.png"

        def get_screen_size(self) -> tuple[int, int]:
            return (1080, 1920)

        def play_card(self, card_index: int, x: float, y: float) -> None:
            pass

        def click(self, x: int, y: int) -> None:
            pass

    mock_bot = MockBot()

    model = ActorCritic(
        bot=mock_bot,
        fixed_input_dim=FIXED_INPUT_DIM,
        num_card_types=NUM_CARD_TYPES,
        card_continuous_feature_dim=CARD_CONTINUOUS_DIM,
        num_discrete_actions=NUM_DISCRETE_ACTIONS,
        continuous_action_dim=CONTINUOUS_ACTION_DIM,
        card_embedding_size=CARD_EMBEDDING_SIZE,
    )

    print("--- Model Architecture ---")
    print(model)
    print("\n" + "=" * 50 + "\n")

    # This example assumes you have a way to create a GameState object.
    # The step method would be called in a loop within your main training script.
    # For a full test, you would need to mock get_full_game_state and calculate_reward.
