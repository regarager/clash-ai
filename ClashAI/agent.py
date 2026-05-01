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

__all__ = ["ActorCritic", "RolloutBuffer"]


class RolloutBuffer:
    def __init__(self):
        self.fixed_inputs = []
        self.card_ids = []
        self.card_continuous_features = []
        self.playable_masks = []
        self.actions = []  # List of (discrete_idx, continuous_action_np)
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.fixed_inputs[:]
        del self.card_ids[:]
        del self.card_continuous_features[:]
        del self.playable_masks[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.rewards)


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
    buffer: RolloutBuffer

    def __init__(
        self,
        fixed_input_dim: int,
        num_card_types: int,
        card_continuous_feature_dim: int,
        num_discrete_actions: int,
        continuous_action_dim: int,
        card_embedding_size: int = 16,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        K_epochs: int = 10,
        entropy_coef: float = 0.01,
    ):
        super(ActorCritic, self).__init__()
        self.buffer = RolloutBuffer()

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef

        self.num_card_slots = num_discrete_actions - 1

        self.fixed_mlp = nn.Sequential(
            nn.Linear(fixed_input_dim, hidden_dim // 2), nn.ReLU()
        )
        self.card_embedding = nn.Embedding(num_card_types + 1, card_embedding_size)

        # The attention mechanism now operates only on the card embeddings (size 16)
        attn_dim = card_embedding_size
        nhead = 4 if attn_dim % 4 == 0 else (2 if attn_dim % 2 == 0 else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=nhead, batch_first=True
        )
        self.cards_mlp = nn.Sequential(nn.Linear(attn_dim, hidden_dim // 2), nn.ReLU())

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
        bot: Bot,
        previous_state: Optional[GameState],
    ) -> GameState:
        """
        Performs one step of the game loop: gets state, selects action,
        calculates reward, and updates the agent.
        """
        print(f"\n--- Agent Step Initiated (Device: {bot.device}) ---")

        # 1. Get current game state
        current_state = get_full_game_state(bot)
        print(f"Current Game State: {current_state}")

        # --- Handle UI Screens (Non-Battle) ---
        if current_state.screen_type == GameScreen.MAIN_PAGE:
            print(
                f"BOT ({bot.device}): Main page detected. Clicking Battle and resetting HandReader."
            )
            reset_hand_reader()
            from .positions import BATTLE

            bot.tap(BATTLE)
            sleep(2)
            return current_state
        elif current_state.screen_type == GameScreen.END_SCREEN:
            print(f"BOT ({bot.device}): End screen detected. Clicking Play Again.")

            from .positions import PLAY_AGAIN

            bot.tap(PLAY_AGAIN)
            sleep(2)
            # If we were in a game, this is a terminal state
            if previous_state and len(self.buffer) > 0:
                self.buffer.is_terminals[-1] = True
            return current_state

        elif current_state.screen_type != GameScreen.GAME_SCREEN:
            from .positions import PLAY_AGAIN

            print(
                f"BOT ({bot.device}): Non-game screen ({current_state.screen_type.name}) - Clicking {PLAY_AGAIN} to attempt skip/dismiss."
            )
            bot.tap((PLAY_AGAIN))
            return current_state

        # --- Handle Active Battle Screen (Agent Actions) ---

        if not current_state.detections:
            print(
                f"AGENT DEBUG ({bot.device}): Skipping step due to zero detections from vision module."
            )
            return current_state  # Skip the rest of the step

        # 2. Ask agent for action
        d_action, c_action, log_prob, state_val = self.select_action(
            current_state.fixed_inputs,
            current_state.card_ids,
            current_state.card_continuous_features,
            current_state.playable_mask,
        )
        print(
            f"Agent selected Discrete Action: {d_action}, Continuous Action: {c_action}"
        )

        # Store state and action info in buffer
        self.buffer.fixed_inputs.append(current_state.fixed_inputs)
        self.buffer.card_ids.append(current_state.card_ids)
        self.buffer.card_continuous_features.append(
            current_state.card_continuous_features
        )
        self.buffer.playable_masks.append(current_state.playable_mask)
        self.buffer.actions.append((d_action, c_action))
        self.buffer.log_probs.append(log_prob)
        self.buffer.state_values.append(state_val)

        # 3. Take action
        if d_action < self.num_card_slots:  # Assuming actions 0-3 are "play card"
            scaled_x = (c_action[0] + 1) / 2
            scaled_y = (c_action[1] + 1) / 2
            print(
                f"Playing card {d_action} at scaled position ({scaled_x:.2f}, {scaled_y:.2f})"
            )
            bot.play_card(d_action, scaled_x, scaled_y)
        else:
            print(f"Agent chose to do nothing (action {d_action}).")

        sleep(1)  # Wait for the game to update after an action
        print(f"AGENT ({bot.device}): Finished waiting.")

        # 4. Calculate reward based on state change
        if previous_state:
            print("AGENT: Calculating reward...")
            reward = calculate_reward(current_state, previous_state)
            self.buffer.rewards.append(reward)
            # Default to False, will be set to True if game ends
            self.buffer.is_terminals.append(False)
            print(f"AGENT: Reward calculated: {reward}")

            # Check if this state is terminal (king tower down)
            if current_state.tower_healths:
                if (
                    current_state.tower_healths.get("enemy-king-tower", 1.0) <= 0.05
                    or current_state.tower_healths.get("ally-king-tower", 1.0) <= 0.05
                ):
                    print("AGENT: King Tower down! Marking as terminal state.")
                    self.buffer.is_terminals[-1] = True
        else:
            # First step of a match, we don't have a reward yet
            # We'll need to handle the offset in the buffer or just skip the first transition
            pass

        # 5. Update the agent periodically
        # PPO usually uses larger batch sizes, e.g., 128 or 256
        if len(self.buffer) >= 128:
            print(
                f"AGENT: Updating agent with PPO (buffer size: {len(self.buffer)})..."
            )
            self.update()
            self.buffer.clear()
            print("AGENT: Agent update complete.")

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
            hand_embeds = self.card_embedding(card_ids)  # [batch, 4, embed_dim]

            # 2. Process Field Unit Features (card_continuous_features can vary)
            if (
                card_continuous_features is not None
                and card_continuous_features.nelement() > 0
            ):
                field_features = card_continuous_features  # [batch, num_units, 4]
                pass

            attn_output, _ = self.attention(hand_embeds, hand_embeds, hand_embeds)
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
                [
                    action_mask,
                    torch.ones(
                        (batch_size, 1), device=action_mask.device, dtype=torch.bool
                    ),
                ],
                dim=1,
            )
            # Apply large negative value to unplayable actions
            discrete_logits[~extended_mask] = -1e10

        discrete_dist = Categorical(logits=discrete_logits)

        # Parameterized continuous action: mean for each card slot
        continuous_params = self.continuous_action_head(combined_features)
        continuous_mean = torch.tanh(continuous_params.view(-1, self.num_card_slots, 2))

        # Action variance (shared across batch, but separate for each card slot)
        action_std = (
            self.continuous_action_log_std.exp().unsqueeze(0).expand_as(continuous_mean)
        )
        continuous_dist = Normal(continuous_mean, action_std)

        state_value = self.critic_head(combined_features)
        return discrete_dist, continuous_dist, state_value

    def select_action(
        self,
        fixed_inputs: torch.Tensor,
        card_ids: torch.Tensor | None,
        card_continuous_features: torch.Tensor | None,
        playable_mask: torch.Tensor | None = None,
    ) -> tuple[int, np.ndarray[Any, Any], torch.Tensor, torch.Tensor]:
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

        with torch.no_grad():
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
            continuous_log_prob = card_dist.log_prob(continuous_action_sampled).sum(
                dim=-1
            )

            continuous_action_np = (
                continuous_action_sampled.squeeze(0).detach().cpu().numpy()
            )
        else:
            # "Do nothing" action has no associated continuous parameter log-prob
            continuous_log_prob = torch.tensor([0.0], device=discrete_action.device)
            continuous_action_np = np.zeros(2)

        total_log_prob = discrete_log_prob + continuous_log_prob

        return (
            discrete_idx,
            continuous_action_np,
            total_log_prob.detach(),
            state_value.detach(),
        )

    def evaluate(
        self,
        fixed_inputs: torch.Tensor,
        card_ids: torch.Tensor,
        card_continuous_features: torch.Tensor,
        playable_masks: torch.Tensor,
        actions: list[tuple[int, np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates a batch of actions for PPO update.
        """
        discrete_dist, continuous_dist, state_values = self.forward(
            fixed_inputs, card_ids, card_continuous_features, playable_masks
        )

        discrete_actions = torch.tensor(
            [a[0] for a in actions], device=fixed_inputs.device
        )
        continuous_actions = torch.tensor(
            [a[1] for a in actions], dtype=torch.float32, device=fixed_inputs.device
        )

        # 1. Discrete Log-Probs and Entropy
        discrete_log_probs = discrete_dist.log_prob(discrete_actions)
        discrete_entropy = discrete_dist.entropy()

        # 2. Continuous Log-Probs and Entropy
        # For continuous actions, we need to map them to the correct card slot distribution
        # This is tricky in batch if each action uses a different card slot.
        # We'll use indexing to pick the right mean/std for each action.

        batch_indices = torch.arange(len(actions), device=fixed_inputs.device)
        # discrete_actions is [batch] with values 0..4
        # We only care about continuous log-probs if discrete_action < 4
        is_play_card = discrete_actions < self.num_card_slots

        # Initialize continuous log probs and entropy
        continuous_log_probs = torch.zeros_like(discrete_log_probs)
        continuous_entropy = torch.zeros_like(discrete_log_probs)

        if is_play_card.any():
            valid_indices = batch_indices[is_play_card]
            valid_card_slots = discrete_actions[is_play_card]

            # Extract relevant mean and std [num_valid, 2]
            card_means = continuous_dist.loc[valid_indices, valid_card_slots]
            card_stds = continuous_dist.scale[valid_indices, valid_card_slots]

            card_dist = Normal(card_means, card_stds)

            # Extract relevant actions [num_valid, 2]
            valid_actions = continuous_actions[is_play_card]

            continuous_log_probs[is_play_card] = card_dist.log_prob(valid_actions).sum(
                dim=-1
            )
            continuous_entropy[is_play_card] = card_dist.entropy().sum(dim=-1)

        total_log_probs = discrete_log_probs + continuous_log_probs
        total_entropy = discrete_entropy + continuous_entropy

        return total_log_probs, state_values.squeeze(), total_entropy

    def update(self) -> None:
        if len(self.buffer) == 0:
            return

        # 1. Convert buffer to tensors
        # Note: card_continuous_features might have different sizes if we used a more complex model,
        # but here we'll assume they are padded or handled by the forward method.
        # For now, we'll just stack what we have.

        device = next(self.parameters()).device

        # We need to handle the case where some entries in card_continuous_features are empty
        # For batching, we'll pad them to the max size in this batch.
        max_units = max([f.shape[0] for f in self.buffer.card_continuous_features])
        padded_continuous_features = []
        for f in self.buffer.card_continuous_features:
            if f.shape[0] < max_units:
                padding = torch.zeros((max_units - f.shape[0], 4), device=f.device)
                padded_continuous_features.append(torch.cat([f, padding], dim=0))
            else:
                padded_continuous_features.append(f)

        old_fixed_inputs = torch.stack(self.buffer.fixed_inputs).to(device)
        old_card_ids = torch.stack(self.buffer.card_ids).to(device)
        old_continuous_features = torch.stack(padded_continuous_features).to(device)
        old_playable_masks = torch.stack(self.buffer.playable_masks).to(device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(device).detach()
        old_state_values = (
            torch.stack(self.buffer.state_values).squeeze().to(device).detach()
        )

        # 2. Calculate Returns and Advantages
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals

        # If the last action wasn't terminal, we might want to bootstrap with the last state value
        # But for simplicity, we'll just use the rewards in the buffer.
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        advantages = returns - old_state_values

        # 3. PPO Update Epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions with current policy
            log_probs, state_values, dist_entropy = self.evaluate(
                old_fixed_inputs,
                old_card_ids,
                old_continuous_features,
                old_playable_masks,
                self.buffer.actions,
            )

            # Ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs)

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Loss components
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * nn.functional.mse_loss(state_values, returns)
            entropy_loss = -self.entropy_coef * dist_entropy.mean()

            loss = actor_loss + critic_loss + entropy_loss

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(
            f"PPO UPDATE: Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Entropy: {dist_entropy.mean().item():.4f}"
        )

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
