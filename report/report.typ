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
    ARW: Proximal Policy Optimization for Real-Time Strategy in Clash Royale
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
      *Date:* May 1, 2026
    ],
  )
]

#v(1em)

#abstract[
  This paper introduces the Autonomous Royale Winner (ARW), a system designed to address the challenges of automating real-time strategy (RTS) play in mobile environments. Specifically targeting Clash Royale, the problem is non-trivial due to the high-dimensional state space and diversity of strategies. Our contribution is a modular orchestration loop that synchronizes Android Debug Bridge (ADB) captures via minicap with a hybrid vision pipeline—utilizing YOLOv11 for unit detection and center-weighted dHash for hand card identification. We implement a Proximal Policy Optimization (PPO) agent with action masking to ensure valid move selection and stable policy convergence. We demonstrate that this architecture achieves autonomous navigation and emergent tactical behavior, significantly improving upon previous iteration baselines.
]

#v(1em)

#show: columns.with(2, gutter: 1em)

= Past Work and Gaps
The evolution of game AI provides the foundation for our research. Traditional
engines like *Stockfish* [1] rely on alpha-beta search and hand-crafted
evaluation functions. While Stockfish achieved superhuman performance in chess,
its logic is domain-specific and fails in environments with hidden information
or continuous action spaces.

In the RTS domain, the *Pbatch/ClashRoyaleBuildABot* [2] project established a
benchmark for mobile automation. It utilized YOLOv5 for unit detection and
image hashing for card recognition. However, as noted in the project's own
roadmap [2], it primarily functions as a "state generator" and lacks a
sophisticated strategic brain, instead using heuristic reward functions for
each unique troop, which cannot easily be scaled.

Furthermore, DeepMind's *AlphaZero* [3] demonstrated that reinforcement
learning (RL) could surpass human-curated knowledge by using self-play.
However, AlphaZero requires perfect information. Similarly, DeepMind's
*AlphaStar* [4] achieved Grandmaster-level play in StarCraft II using deep
reinforcement learning, but relied on a direct API interface rather than
visual perception—a luxury unavailable for mobile games. Our work bridges the
gap between the raw state-generation of *Build-A-Bot* and the deep strategic
learning demonstrated by *AlphaZero* and *AlphaStar*, by applying a hybrid
Actor-Critic model to the partially-observable, real-time environment of mobile
RTS.

= Proposal and Preliminary Hypothesis
We propose that a modular vision-to-RL pipeline will demonstrate superior sample efficiency compared to end-to-end pixel-based models. By decoupling feature extraction from policy optimization, the agent can bypass the "feature-discovery" phase.

- *Hypothesis:* A Proximal Policy Optimization (PPO) agent utilizing a Multi-head Attention mechanism [7] and action masking will effectively derive spatial relationships and resource management strategies, leading to higher win rates than unmasked or heuristic baselines.
- *Plan of Study:* The project has progressed from architectural validation to autonomous training. The current stage involves long-term PPO optimization using parallel environment instances to refine strategic placement.
- *Validation:* Validated if the agent achieves a positive trophy trend and maintains a win rate above 50% in controlled arena environments.
- *Falsification:* Failure to converge on a stable policy or inability to outperform a random-action baseline with action masking.

= Design Justification
We transitioned from A2C to the *Proximal Policy Optimization (PPO)* framework [5]. Unlike standard actor-critic methods, PPO uses a clipped objective function to prevent updates from being too large, which is critical for stability in the volatile environment of a real-time mobile game where state transitions can be sudden.

To handle the "action-validity" problem, we implemented *Action Masking*. In Clash Royale, cards have varying elixir costs. By masking unplayable cards in the policy output based on real-time elixir detection, we force the agent to explore only legal moves, significantly accelerating the convergence of the neural network.

The vision system was upgraded to a hybrid pipeline. While YOLOv11 remains the engine for dynamic unit detection, we identified that hand card recognition requires higher precision to avoid "hallucinations" (detecting units in the UI). We implemented a *64x64 invariant dHash (Difference Hash)* for hand slots. To achieve translation invariance, the system employs *Sobel edge detection* and *Otsu thresholding* to isolate the card's visual content before generating a *4096-bit hash*. To account for UI jitter, we implement *Jitter-Robust matching*, which samples five spatial offsets per slot. Cards are identified if they fall within a Hamming distance threshold of 1500 bits (approx. 36% error budget), ensuring near-perfect accuracy against a known whitelist.

For performance, we integrated *minicap* for screen capture, reducing capture latency from $approx 200$ms to $approx 30$ms. This allows the agent to react to fast-moving units with much higher fidelity.

= Experimentation Methodology
The system operates via `main.py` and `ClashAI/core.py` in an autonomous loop.
+ *Navigation:* `bot.py` uses template matching to detect "Battle" buttons and other similar prompts, allowing for unattended training sessions across multiple matches.
+ *Perception:* `vision.py` captures the screen via ADB/minicap. It uses YOLOv11 (`best.pt`) for unit detection and `HandReader` (dHash) for card identification.
+ *Decision*: An Actor-Critic model, trained via the PPO algorithm, takes the game state as input (card positions, elixir, etc.) and outputs in a hybrid action space, consisting of the card played and its position.
+ *Execution:* Actions are dispatched via the ADB shell, where inputs like taps can be done using shell commands.

== Assumptions
- *State Consistency:* We assume the visual features extracted (unit positions, elixir, hand) provide a sufficient Markov state for the agent.
- *Network Latency:* We assume the local YOLO inference time is negligible compared to the game's 1s decision interval.
- *Restricted Deck:* Training is conducted using a standardized deck to minimize variance and maximize usability given the limited time.

#figure(
  image("architecture-diagram.png", width: 100%),
  caption: [Architecture of the Project],
)

= Results
The agent was evaluated over a series of 25 competitive matches against human opponents at the 1000-trophy level. To ensure consistency, the agent utilized a standardized starter deck. The results are recorded below.
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": chart

#figure(
  table(
    columns: 10,
    inset: 4pt,
    align: center,
    [*1*], [*2*], [*3*], [*4*], [*5*], [*6*], [*7*], [*8*], [*9*], [*10*],
    [W], [W], [W], [L], [L], [W], [W], [W], [L], [W],
    [*11*], [*12*], [*13*], [*14*], [*15*], [*16*], [*17*], [*18*], [*19*], [*20*],
    [W], [W], [W], [W], [W], [W], [W], [L], [W], [L],
    [*21*], [*22*], [*23*], [*24*], [*25*], [], [], [], [], [],
    [W], [L], [W], [W], [W], [], [], [], [], [],
  ),
  caption: [Chronological Match Results (25 Matches)],
)

#let data = (
  ("Wins",   19),
  ("Losses", 6),
)

#cetz.canvas({
  chart.piechart(
    data,
    value-key: 1,
    label-key: 0,
    radius: 3,
    stroke: none,
    slice-style: (rgb("#2ecc71"), rgb("#e74c3c")),  // discrete green, red
    inner-radius: 1.2,
    outset: 0,
    inner-label: (content: (value, label) => [*#str(value)*], radius: 100%),
    outer-label: (content: (value, label) => [#label], radius: 115%),
  )
})

Constructing a binomial confidence interval using the Clopper-Pearson method, we find that the 95% confidence interval for the winrate of the agent is $(0.549, 0.906)$. Therefore, we conclude that there is evidence that the agent is able to play at the 1000 trophy level, i.e. advance to the next arena.

= Analysis
The implementation of *Action Masking* has significantly addressed the previously observed *Elixir Leaking* issue. By preventing the agent from selecting cards that exceed current elixir reserves, the model now exhibits a notably *aggressive playstyle*. Rather than holding cards, the PPO agent tends to spend elixir as soon as it is available, prioritizing offensive momentum over resource conservation.

A significant emergent behavior is the *Dual-Lane Push*. The Multi-head Attention layer [7] allows the agent to recognize opportunities to split pressure across both towers. This often overwhelms opponents in the 1000-trophy range who struggle to defend two lanes simultaneously.

However, this aggression comes at the cost of being *weaker defensively*. Because the agent is biased toward playing cards offensively as soon as the action mask allows, it often lacks the elixir reserves necessary to react to sudden counter-pushes. At the agent's current trophy level (1000), this playstyle is sufficient for achieving victory over other players, but at higher levels this strategy will likely fail.

= Limitations
While the PPO-based architecture demonstrates clear improvements over prior iterations, several constraints remain that bound the agent's current capabilities:

- *Decision Latency:* The approximately 1-second perception-decision loop—composed of minicap capture, YOLOv11 inference, and post-action delay—is adequate for offensive card deployment but too slow for precise reactive maneuvers. Defensive actions such as troop "kiting" or spell placement against fast-moving threats (e.g., Hog Rider, Prince) require sub-second timing that the current pipeline cannot reliably achieve.

- *Perception Accuracy:* The YOLOv11 object detection model, while selected for inference speed over raw accuracy, introduces occasional classification errors on visually similar units. These misclassifications propagate into the agent's state representation, causing suboptimal decisions. Furthermore, the current object detection model scales poorly given the huge amount of cards in the game, in addition to variations of base cards such as Heroes and Evolutions. Addressing this through direct state acquisition (see Future Work §1) may prove more robust than incremental model refinement.

- *Shifting Object:* While not a limitation of the agent itself, it can be difficult to accurately evaluate the agent's skill, as other variables such as card levels and opponent skill can influence the outcomes of a match.

= Conclusion
The project successfully demonstrates that a modular PPO backend, combined with a specialized vision pipeline, can autonomously navigate the complexities of a mobile RTS. The transition from A2C to PPO with action masking produced measurable improvements in resource utilization and eliminated the elixir leakage observed in prior iterations. The dHash-based hand reader provides near-perfect card identification, solving the hallucination problem inherent in pure object detection approaches.

However, the findings also reveal that solving the action-validity problem introduces new challenges in strategic balance. The agent's emergent aggression, while tactically effective at low trophy ranges, underscores that reinforcement learning in resource-constrained environments requires careful reward shaping to prevent policy collapse toward extreme behaviors. The current limitations in decision latency and perception accuracy represent the primary barriers to higher-level competitive play.

= Future Work
If granted additional time, we would pursue several directions, each addressing specific limitations identified above:

1. *Memory-based State Detection:* Extract game state directly from the game's runtime memory rather than through computer vision. This would provide perfectly accurate state information and eliminate both classification errors and the dependency on fixed UI layouts (addressing Limitations §2). However, this approach requires significant reverse engineering effort.

2. *Recurrent Policies:* Integrate Long Short-Term Memory (LSTM) layers into the policy network. A recurrent architecture would allow the agent to track opponent card cycles across time steps, enabling predictive defensive play that compensates for the current decision latency (addressing Limitations §1).

3. *Advanced Reward Shaping:* Implement rewards based on tower health differentials and elixir efficiency ratios. Explicitly penalizing "overcommitment" (spending elixir without corresponding tower damage) would counteract the aggressive bias introduced by action masking.

4. *Increased Game Information:* Incorporate Evolution and Hero cards into the action space and detection pipeline, expanding the agent's strategic vocabulary to match the full scope of modern Clash Royale gameplay.

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
  [Schulman, J., et al. "Proximal Policy Optimization Algorithms." _arXiv preprint arXiv:1707.06347_, 2017.],
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
