import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np

class ActorCritic(nn.Module):
    """
    An Actor-Critic model for a hybrid action space (discrete + continuous)
    and mixed input types, updated to properly handle categorical card IDs.

    This model is designed for a reinforcement learning environment where the agent
    must choose a discrete action and a continuous action.

    The architecture uses a shared backbone to process three types of inputs:
    1. A fixed-size vector of numerical features (e.g., tower HP, elixir).
    2. Categorical IDs for each card on screen (e.g., knight=0, archer=1).
    3. A vector of continuous features for each card (e.g., x, y, hp_ratio).

    The model has three outputs:
    1. A probability distribution for the discrete actions (Actor's policy).
    2. A probability distribution for the continuous actions (Actor's policy).
    3. An estimated state value (Critic's value).
    """
    def __init__(self, fixed_input_dim, num_card_types, card_continuous_feature_dim,
                 num_discrete_actions, continuous_action_dim, card_embedding_size=16, hidden_dim=256):
        """
        Initializes the model layers.

        Args:
            fixed_input_dim (int): Dimensionality of the fixed numerical input vector.
            num_card_types (int): The number of unique card types for the embedding layer.
            card_continuous_feature_dim (int): Dimensionality of the continuous feature vector for each card.
            num_discrete_actions (int): The number of possible discrete actions.
            continuous_action_dim (int): Dimensionality of the continuous action space (e.g., 2 for x, y).
            card_embedding_size (int): The size of the dense embedding for each card type.
            hidden_dim (int): The size of the hidden layers.
        """
        super(ActorCritic, self).__init__()

        # --- Input Processing Layers ---

        # 1. MLP for fixed numerical inputs
        self.fixed_mlp = nn.Sequential(
            nn.Linear(fixed_input_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 2. Embedding layer for categorical card IDs
        self.card_embedding = nn.Embedding(num_card_types, card_embedding_size)
        
        # The total dimension for one card's features after embedding and concatenation
        card_feature_dim = card_embedding_size + card_continuous_feature_dim
        
        # 3. Self-Attention for all card features
        # nhead must be a divisor of the feature dimension
        nhead = 4 if card_feature_dim % 4 == 0 else (2 if card_feature_dim % 2 == 0 else 1)
        self.attention = nn.MultiheadAttention(embed_dim=card_feature_dim, num_heads=nhead, batch_first=True)
        self.cards_mlp = nn.Sequential(
            nn.Linear(card_feature_dim, hidden_dim // 2),
            nn.ReLU()
        )

        combined_feature_dim = hidden_dim

        # --- Actor Heads ---
        self.discrete_action_head = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_discrete_actions)
        )
        self.continuous_action_head = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, continuous_action_dim)
        )
        self.continuous_action_log_std = nn.Parameter(torch.zeros(continuous_action_dim))

        # --- Critic Head ---
        self.critic_head = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, fixed_inputs, card_ids, card_continuous_features):
        """
        Defines the forward pass of the model.

        Args:
            fixed_inputs (torch.Tensor): Shape (batch_size, fixed_input_dim).
            card_ids (torch.LongTensor): Shape (batch_size, num_cards).
            card_continuous_features (torch.Tensor): Shape (batch_size, num_cards, card_continuous_feature_dim).

        Returns:
            (Categorical, Normal, torch.Tensor): Distributions for actions and the state value.
        """
        fixed_features = self.fixed_mlp(fixed_inputs)

        if card_ids is not None and card_ids.nelement() > 0:
            card_embeds = self.card_embedding(card_ids)
            card_full_features = torch.cat([card_embeds, card_continuous_features], dim=-1)
            
            attn_output, _ = self.attention(card_full_features, card_full_features, card_full_features)
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

    def select_action(self, fixed_inputs, card_ids, card_continuous_features):
        """
        Selects an action by sampling from the policy distributions.
        """
        if fixed_inputs.dim() == 1:
            fixed_inputs = fixed_inputs.unsqueeze(0)
        if card_ids is not None and card_ids.nelement() > 0:
            if card_ids.dim() == 1:
                card_ids = card_ids.unsqueeze(0)
            if card_continuous_features.dim() == 2:
                card_continuous_features = card_continuous_features.unsqueeze(0)

        discrete_dist, continuous_dist, _ = self.forward(fixed_inputs, card_ids, card_continuous_features)

        discrete_action = discrete_dist.sample()
        continuous_action = continuous_dist.sample()

        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)

        return (
            discrete_action.item(),
            continuous_action.squeeze(0).detach().cpu().numpy(),
            discrete_log_prob.item(),
            continuous_log_prob.item()
        )

    def get_value(self, fixed_inputs, card_ids, card_continuous_features):
        """
        Gets the value of a state from the critic.
        """
        _, _, state_value = self.forward(fixed_inputs, card_ids, card_continuous_features)
        return state_value


if __name__ == '__main__':
    # --- Example Usage ---
    FIXED_INPUT_DIM = 3
    NUM_CARD_TYPES = 110 # Total number of unique cards in the game
    CARD_CONTINUOUS_DIM = 4 # [x, y, hp_ratio, is_enemy]
    NUM_DISCRETE_ACTIONS = 5
    CONTINUOUS_ACTION_DIM = 2
    CARD_EMBEDDING_SIZE = 16

    model = ActorCritic(
        fixed_input_dim=FIXED_INPUT_DIM,
        num_card_types=NUM_CARD_TYPES,
        card_continuous_feature_dim=CARD_CONTINUOUS_DIM,
        num_discrete_actions=NUM_DISCRETE_ACTIONS,
        continuous_action_dim=CONTINUOUS_ACTION_DIM,
        card_embedding_size=CARD_EMBEDDING_SIZE
    )

    print("--- Model Architecture ---")
    print(model)
    print("\n" + "="*50 + "\n")

    # --- Test with Dummy Data ---
    BATCH_SIZE = 4
    NUM_CARDS_ON_SCREEN = 6

    fixed_data = torch.randn(BATCH_SIZE, FIXED_INPUT_DIM)
    # Card IDs should be Long type for embedding layer
    card_ids_data = torch.randint(0, NUM_CARD_TYPES, (BATCH_SIZE, NUM_CARDS_ON_SCREEN))
    card_continuous_data = torch.randn(BATCH_SIZE, NUM_CARDS_ON_SCREEN, CARD_CONTINUOUS_DIM)

    print(f"--- Testing Forward Pass with Batch Size: {BATCH_SIZE}, Num Cards: {NUM_CARDS_ON_SCREEN} ---")
    
    discrete_dist, continuous_dist, value = model(fixed_data, card_ids_data, card_continuous_data)

    print(f"Discrete action distribution (sample shape): {discrete_dist.sample().shape}")
    print(f"Continuous action distribution mean (shape): {continuous_dist.mean.shape}")
    print(f"Critic value output (shape): {value.shape}")
    print("\n" + "="*50 + "\n")
    
    # --- Test action selection for a single state ---
    print("--- Testing Action Selection for a Single State ---")
    
    single_fixed_data = torch.randn(FIXED_INPUT_DIM)
    single_card_ids = torch.randint(0, NUM_CARD_TYPES, (3,)) # 3 cards on screen
    single_card_continuous = torch.randn(3, CARD_CONTINUOUS_DIM)

    discrete_act, continuous_act, _, _ = model.select_action(
        single_fixed_data, single_card_ids, single_card_continuous
    )
    
    print(f"Selected Discrete Action: {discrete_act}")
    print(f"Selected Continuous Action (x, y): {continuous_act}")
    print("\n" + "="*50 + "\n")
    
    # --- Test with no cards on screen ---
    print("--- Testing with No Cards on Screen ---")
    
    no_card_ids = torch.empty(0, dtype=torch.long)
    no_card_continuous = torch.empty(0) 
    
    discrete_act, continuous_act, _, _ = model.select_action(
        single_fixed_data, no_card_ids, no_card_continuous
    )
    
    print(f"Selected Discrete Action (no cards): {discrete_act}")
    print(f"Selected Continuous Action (no cards): {continuous_act}")
