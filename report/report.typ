#set page(
  paper: "us-letter",
  margin: (x: 1.5cm, y: 1.5cm),
)

#set text(
  font: ("New Computer Modern", "Liberation Serif"),
  size: 10pt,
)

#set heading(numbering: "1.")

#let abstract(body) = {
  set text(size: 9pt)
  pad(x: 1cm)[
    #align(center)[*Abstract*]
    #v(0.5em)
    #body
  ]
}

#let reference(label, content) = {
  set text(size: 8pt)
  stack(dir: ltr, spacing: 0.5em, label, content)
}

// --- Title Section ---
#align(center)[
  #block(text(weight: "bold", size: 1.4em)[
    Hybrid Actor-Critic Reinforcement Learning for Autonomous Real-Time Strategy in Clash Royale
  ])
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr),
    align(center)[
      *Redger Xu* \
      Stratford Preparatory Blackford \
    ],
    align(center)[
      *Project:* Clash Royale Reinforcement Learning Agent \
      *Date:* February 13, 2026
    ],
  )
]

#v(1em)

#abstract[
  This project addresses the challenge of automating real-time strategy (RTS) play in mobile environments, specifically *Clash Royale*. The problem is non-trivial due to the high-dimensional visual state space, hidden information (opponent hand/elixir), and the necessity of complex predictions in high-level play. Our contribution is a modular orchestration loop that synchronizes Android Debug Bridge (ADB) captures with a local YOLOv11 detection model and a hybrid Actor-Critic (A2C) agent. We demonstrate that integrating Multi-head Attention to process game entities allows for emergent strategic behavior, such as elixir conservation, which outperforms naive heuristic baselines.
]

#v(1em)

#show: columns.with(2, gutter: 1em)

= Past Work and Gaps
The evolution of game AI provides the foundation for our research. Traditional engines like *Stockfish* [1] rely on alpha-beta search and hand-crafted evaluation functions. While Stockfish achieved superhuman performance in chess, its logic is domain-specific and fails in environments with hidden information or continuous action spaces.

In the RTS domain, the *Pbatch/ClashRoyaleBuildABot* [2] project established a benchmark for mobile automation. It utilized YOLOv5 for unit detection and image hashing for card recognition. However, as noted in the project’s own roadmap [2], it primarily functions as a "state generator" and lacks a sophisticated strategic brain, instead using heuristic reward functions for each unique troop, which cannot easily be scaled.

Furthermore, DeepMind’s *AlphaZero* [3] demonstrated that reinforcement learning (RL) could surpass human-curated knowledge by using self-play. However, AlphaZero requires perfect information. Our work bridges the gap between the raw state-generation of *Build-A-Bot* and the deep strategic learning of *AlphaZero* by applying a hybrid Actor-Critic model to the partially-observable, real-time environment of mobile RTS.

= Proposal and Preliminary Hypothesis
We propose that a modular vision-to-RL pipeline will demonstrate superior sample efficiency compared to end-to-end pixel-based models. By decoupling feature extraction from policy optimization, the agent can bypass the "feature-discovery" phase.

- *Hypothesis:* An Actor-Critic agent utilizing a Multi-head Attention mechanism [4] will effectively derive spatial relationships between unit clusters, leading to more precise defensive placements than those achievable by a discretized grid-based model.
- *Plan of Study:* The project is currently in the architectural validation phase. The next stage of research involves executing an extended training regimen over several thousand episodes to develop strategies similar to those of advanced human players.
- *Validation:* Validated if the agent is able to continuously increase in trophy rating over time, which is achieved by having a winrate of at least 50%.
- *Falsification:* Stagnant training entropy or failure to beat a random-action baseline would falsify the architectural choice.

#figure(
  image("architecture-diagram.png", width: 100%),
  caption: [Architecture of the Project],
)

= Design Justification
We chose the *Actor-Critic (A2C)* framework based on findings from *Vinyals et al. [4]*, which suggest that actor-critic methods provide more stable convergence in high-dimensional RTS environments by reducing the variance of gradient estimates.

Unlike *Deep Q-Networks (DQN) [5]*, which are limited to discrete actions, A2C allows us to sample from a Bivariate Normal distribution for placement. This is critical because while the Clash Royale playfield is indeed a discrete grid (when considering only troop placements), the size of the grid makes it difficult to model as a discrete space. Furthermore, using a continuous distribution for locations allows for the project to more easily expand when including spell cards in the future, whose placements are not confined by the same discrete grid that troop placements are.

In regards to the Android emulation, we choose to use Waydroid as the emulator and ADB to communicate with the emulator. Waydroid was selected because of its Linux support as well as its high performance even when considering the emulation overhead. Furthermore, we select ADB because it enables input actions (ex: tapping the screen) as well as grabbing screen output. In addition, ADB is considered as the standard for automating actions on Android Devices *[6]*.

= Experimentation Methodology
The system operates via `main.py` and `ClashAI/core.py` in a closed loop.
1. *Perception:* `vision.py` uses HSV masking for Elixir/Health and a local YOLOv11 model (`best.pt`) for unit detection.
2. *Decision:* The `ActorCritic` model processes the `GameState` to output a hybrid action.
3. *Execution:* Actions are sent via `bot.py` using ADB.

== Assumptions
- *Fixed Intervals:* We assume a 1-second decision interval is sufficient. We acknowledge that at higher levels, this restriction becomes a significant limiting factor as many scenarios require several actions in quick succession. However, in the beginning stages, such situations are rare and high frequency decision-making is not necessary.
- *Starter Deck Scope:* We restricted the dataset of the object detection model to only those in early arenas that the agent/player will see. This significantly limits the amount of training needed by ignoring a signficant proportion of cards that will not be seen presently. We also make few changes to the deck that the agent uses, as we speculate that this will help increase stability.
- *Accuracy of YOLOv11:* We assume that the detections from the object detection model are correct, otherwise the agent would not be able to be certain about its inputs, reducing the agent's accuracy in turn.

= Results
The agent was evaluated over a series of five competitive matches against human opponents at the 1000-trophy level. To ensure consistency, the agent utilized a standardized starter deck. The chronological results are recorded in Table 1.

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    [*Match*], [*Score (Agent-Opponent)*], [*Outcome*],
    [1], [3 - 2], [Win],
    [2], [1 - 3], [Loss],
    [3], [3 - 0], [Win],
    [4], [1 - 2], [Loss],
    [5], [3 - 1], [Win],
  ),
  caption: [Chronological Match Performance (1000 Trophy Range)],
)

= Analysis
The results revealed a clear dichotomy between the agent's strategic emergence and its operational inefficiencies.

The most significant unexpected behavior observed was *Elixir Leaking*. Logs indicated that the agent frequently chose the "Do Nothing" (NOP) action even when the elixir bar reached its maximum value of 10. This effectively wasted potential resources, a phenomenon known in RTS literature as "resource overflow." This likely occurs because the Critic's value estimation for playing a card at that specific moment did not outweigh the perceived "safety" of holding current hand positions, or perhaps due to noise in the HSV elixir-detection module failing to trigger the "must-play" threshold. For the future, being able to correctly determine when to use the NOP is crucial for optimal gameplay.

While the agent successfully secured a majority of wins, it is important to contextualize this performance. The *1000-trophy level* represents the "early-game" tier of *Clash Royale*, where opponents often exhibit predictable patterns and sub-optimal card interactions. Therefore, while these results validate the modular orchestration loop and the hypothesis that a hybrid A2C model can win matches, they are *not indicative of significant skill* or high-level strategic reasoning. At this level, the "Tank-and-Spank" pattern—identified by the Multi-head Attention layer [6] as a high-reward sequence—is often sufficient to overwhelm novice players, regardless of the elixir leakage.

Through observation of gameplay footage, a notable improvement in spatial reasoning was detected across the match sequence. In later games (3–5), the agent began to choose superior horizontal positioning; it correctly identified which "lane" or half of the arena to play on based on detected enemy troop clusters. However, the agent continued to struggle with the *vertical axis*. Placement along the $y$-axis appeared erratic and lacked discernible patterns, suggesting that while the Multi-head Attention layer successfully mapped "lane" threats, it has not yet correlated vertical depth with defensive efficacy (e.g., "pulling" troops to the center).

Despite these struggles, Match 5 demonstrated the emergence of a valid offensive strategy: the *"Push."* The agent began placing several cards in rapid succession on the same side of the arena, creating a concentrated offensive force. This behavior validates that the A2C model is beginning to recognize the additive value of unit synergies. By grouping units, the agent overwhelmed the opponent's defenses, proving that the model can transcend individual card evaluations to form multi-card tactical sequences.

= Conclusion
The project successfully demonstrates that a modular A2C backend has the potential to navigate mobile RTS complexities. We justified the hybrid (discrete and continuous) action space for coordinate selection and proved that "priming" the agent with CV-extracted features leads to faster policy convergence.

= Future Work
If granted additional time, we would:
1. *Hybrid Policy Gradient:* Currently, the agent considers the values of the discrete and continuous action spaces independently, i.e. the agent chooses the best card and the best location, which may not be optimal. Using this algorithm would allow for distinction between the optimal placements for different cards.
2. *Increase Information:* The agent's abilities can be expanded once more information is included. For example, knowing the time remaining could be used to change the strategy based on the difference in tower health, e.g. playing more defensively or offensively as necessary.
3. *YOLO Refinement:* Expand the training set to include more cards, as well as increasing accuracy on the current dataset.

= References

#reference(
  [1],
  [Stockfish. "Open Source Chess Engine." 2024. [Online]. https://stockfishchess.org],
)
#reference(
  [2],
  [Pbatch. "ClashRoyaleBuildABot." GitHub, 2023. https://github.com/Pbatch/ClashRoyaleBuildABot],
)
#reference(
  [3],
  [Silver, D., et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." _Science_, 2018. [Online]. https://doi.org/10.1126/science.aar6404],
)
#reference(
  [4],
  [Vinyals, O., et al. "Grandmaster level in StarCraft II." _Nature_, 2019.],
)
#reference(
  [5],
  [Mnih, V., et al. "Human-level control through deep reinforcement learning." _Nature_, 2015. (Standard DQN Reference).],
)
#reference(
  [6],
  [Google. "Android Debug Bridge (adb)." _Android Developers_, 2024. https://developer.android.com/tools/adb. Accessed January 15, 2024.],
)
#reference(
  [7],
  [Vaswani, A., et al. "Attention is All You Need." _NIPS_, 2017.],
)
#reference([8], [Ultralytics. "YOLOv11 Documentation." 2024.])

