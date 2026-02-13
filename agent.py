from typing import Any, Optional
from time import sleep 

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing_extensions import override

from vision import get_full_game_state, calculate_reward
from bot import Bot
from game_state import GameState

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

        self.fixed_mlp = nn.Sequential(
            nn.Linear(fixed_input_dim, hidden_dim // 2), nn.ReLU()
        )
        self.card_embedding = nn.Embedding(num_card_types, card_embedding_size)
        card_feature_dim = card_embedding_size + card_continuous_feature_dim
        nhead = 4 if card_feature_dim % 4 == 0 else (2 if card_feature_dim % 2 == 0 else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=card_feature_dim, num_heads=nhead, batch_first=True
        )
        self.cards_mlp = nn.Sequential(
            nn.Linear(card_feature_dim, hidden_dim // 2), nn.ReLU()
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
            nn.Linear(hidden_dim, continuous_action_dim),
        )
        self.continuous_action_log_std = nn.Parameter(
            torch.zeros(continuous_action_dim)
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
        print("\\n--- Agent Step Initiated ---")

        # 1. Get current game state
        current_state = get_full_game_state(self.bot)
        print(f"Current Game State: {current_state}")

        if previous_state is None:
            return current_state
            
        # 2. Ask agent for action
        d_action, c_action = self.select_action(
            current_state.fixed_inputs,
            current_state.card_ids,
            current_state.card_continuous_features,
        )
        print(f"Agent selected Discrete Action: {d_action}, Continuous Action: {c_action}")

        # If no objects are detected, force "do nothing" action
        if current_state.card_ids.nelement() == 0 and d_action < 4:
            print("AGENT INFO: No objects detected. Overriding action to 'do nothing' (action 4).")
            d_action = 4 # Force "do nothing" action
            
        # 3. Take action
        if d_action < 4:  # Assuming actions 0-3 are "play card"
            scaled_x = (c_action[0] + 1) / 2
            scaled_y = (c_action[1] + 1) / 2
            print(f"Playing card {d_action} at scaled position ({scaled_x:.2f}, {scaled_y:.2f})")
            self.bot.play_card(d_action, scaled_x, scaled_y)
        else:
            print("Agent chose to do nothing (action 4).")

        sleep(2)  # Wait for the game to update after an action
        print("AGENT: Finished waiting.")

        # 4. Calculate reward based on state change
        print("AGENT: Calculating reward...")
        reward = calculate_reward(current_state, previous_state)
        self.rewards.append(reward)
        print(f"AGENT: Reward calculated: {reward}")

        # 5. Update the agent
        print("AGENT: Updating agent (training step)...")
        self.update()
        print("AGENT: Agent update complete.")

        return current_state

    @override
    def forward(
        self,
        fixed_inputs: torch.Tensor,
        card_ids: torch.Tensor | None,
        card_continuous_features: torch.Tensor | None,
    ) -> tuple[Categorical, Normal, torch.Tensor]:
        fixed_features = self.fixed_mlp(fixed_inputs)

        if card_ids is not None and card_ids.nelement() > 0:
            card_embeds = self.card_embedding(card_ids)
            if card_continuous_features is not None and card_continuous_features.nelement() > 0:
                card_full_features = torch.cat(
                    [card_embeds, card_continuous_features], dim=-1
                )
            else:
                card_full_features = card_embeds
            
            attn_output, _ = self.attention(
                card_full_features, card_full_features, card_full_features
            )
            card_features_agg = attn_output.mean(dim=1)
            card_features = self.cards_mlp(card_features_agg)
        else:
            card_features = torch.zeros_like(fixed_features)

        combined_features = torch.cat([fixed_features, card_features], dim=1)

        discrete_logits = self.discrete_action_head(combined_features)
        discrete_dist = Categorical(logits=discrete_logits)

        continuous_mean = torch.tanh(self.continuous_action_head(combined_features))
        action_std = self.continuous_action_log_std.exp().expand_as(continuous_mean)
        continuous_dist = Normal(continuous_mean, action_std)

        state_value = self.critic_head(combined_features)
        return discrete_dist, continuous_dist, state_value

    def select_action(
        self,
        fixed_inputs: torch.Tensor,
        card_ids: torch.Tensor | None,
        card_continuous_features: torch.Tensor | None,
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

        discrete_dist, continuous_dist, state_value = self.forward(
            fixed_inputs, card_ids, card_continuous_features
        )

        discrete_action = discrete_dist.sample()
        continuous_action = continuous_dist.sample()

        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        self.log_probs.append(discrete_log_prob + continuous_log_prob)
        self.state_values.append(state_value)

        return (
            int(discrete_action.item()),
            continuous_action.squeeze(0).detach().cpu().numpy(),
        )

    def update(self, gamma: float = 0.99) -> None:
        if not self.rewards:
            return

        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        
        # If all rewards are the same (e.g., all zeros), stddev will be 0.
        # This can lead to NaNs in normalized returns and subsequent loss calculation.
        # Skip update if rewards are essentially constant, as there's no learning signal.
        if len(rewards) < 2 or rewards.std() < 1e-6: # Check if std is too small or only one reward
            print(f"AGENT WARNING: Skipping update due to constant or insufficient rewards. Stddev: {rewards.std().item()}")
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
        if returns_std == 0: # Ensure we don't divide by zero if all returns are identical
            returns_tensor = returns_tensor - returns_tensor.mean() # Just center it
        else:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_std + 1e-9)

        advantage = returns_tensor - state_values.detach()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(state_values, returns_tensor)
        loss = actor_loss + 0.5 * critic_loss

        # Check for NaNs in loss before backpropagation
        if torch.isnan(loss):
            print("AGENT WARNING: NaN detected in loss. Skipping backpropagation and resetting buffers.")
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
    ) -> torch.Tensor:
        _, _, state_value = self.forward(
            fixed_inputs, card_ids, card_continuous_features
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
        def screenshot(self) -> str: return "mock_screenshot.png"
        def get_screen_size(self) -> tuple[int, int]: return (1080, 1920)
        def play_card(self, card_index: int, x: float, y: float) -> None: pass
        def click(self, x: int, y: int) -> None: pass

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