# PRD.md — Product Requirements Document: RL-BLOG-ANIMATION

## 1. Product Vision & Executive Summary

**RL-BLOG-ANIMATION** is a visual and mathematical curriculum designed to demystify Reinforcement Learning (RL). By marrying the visual storytelling of 3Blue1Brown's ManimGL engine with formal mathematical rigor and executable code traces, the project transforms abstract RL equations into intuitive, dynamic animations accompanied by publication-grade blog explainers.

---

## 2. Target Audience & User Stories

### Target Personas
1. **The Visual ML Engineer:** Understands supervised learning, wants to master RL (bandits, MDPs, policy gradients, RLHF), but finds pure mathematical notation without dynamic visual traces opaque.
2. **The University Student / Researcher:** Needs clear derivations of regret bounds, Bellman equations, and value function convergence to supplement course textbooks (Sutton & Barto).
3. **The AI Educator / Technical Writer:** Wants clean, high-resolution visual assets, diagrams, and reproducible animations for lectures and technical blog posts.

### User Stories
- **As a learner**, I want to see how an agent's confidence bounds or empirical estimates evolve step-by-step alongside the true hidden parameters, so that I can develop an intuitive mental model of the exploration-exploitation dilemma.
- **As a developer**, I want to run a single command and render a specific sub-scene or snapshot in seconds, so that I can modify parameters and see the visual effects immediately.
- **As a reader**, I want a self-contained markdown article with matching LaTeX equations, tabular comparisons, and embedded video clips for each concept.

---

## 3. Product Scope & Feature Boundaries

### In-Scope (Core Deliverables)
- **Modular Topic Folders:** Each RL concept resides in its own isolated module (e.g. `Multiarmbandit/`, future `MDP/`, `QLearning/`).
- **Triple-Scene Architecture:** Every module must provide at least:
  1. `*Definition`: Formal mathematical formulation, notation, and conceptual setup.
  2. `*UsageExample`: A concrete, seed-fixed algorithmic simulation running over time with dynamic data visualization (e.g., headline click-through rates).
  3. `*Explainer`: An integrated composition combining definition and simulation for a complete video lesson.
- **Deep-Dive Article (`README.md`):** Complete written explanation, practical use cases, limitation analysis, and algorithm pseudocode.
- **Config & Asset Pipelines:** Project-level overrides for fonts, TeX templates, and automated MP4 exports.

### Out-of-Scope (Explicit Non-Goals)
- ❌ **Web Application Backend / User Accounts:** This is an open-source educational repository and animation production engine, not a SaaS platform.
- ❌ **Massive Distributed RL Training:** We simulate illustrative environments and tabular/analytical benchmarks, not multi-GPU cluster training.
- ❌ **Complex 3D Physics Simulation:** Primary focus is clear 2D mathematical vector graphics via ManimGL, not heavy MuJoCo/Isaac Gym environments.

---

## 4. Curriculum Roadmap

| Module | Status | Core Concepts | Visual Scenes |
| :--- | :---: | :--- | :--- |
| **01. Multi-Armed Bandit** | 🚀 **Completed (v0.1.0)** | Exploration vs Exploitation, Epsilon-Greedy, Regret, UCB, Thompson Sampling | `BanditDefinition`, `BanditUsageExample`, `BanditExplainer` |
| **02. Markov Decision Processes (MDP)** | 📋 **Planned (v0.2.0)** | States, Actions, Transition Probabilities $P(s' \mid s, a)$, Reward Functions, Discount Factor $\gamma$ | `MDPGraphScene`, `DiscountHorizonScene`, `MDPExplainer` |
| **03. Bellman Equations & DP** | 📋 **Planned (v0.3.0)** | Bellman Expectation & Optimality, Value Iteration, Policy Iteration, Gridworld | `GridworldValueScene`, `BellmanBackupTree`, `DPConvergenceExplainer` |
| **04. Model-Free TD & Q-Learning** | 📋 **Planned (v0.4.0)** | TD Error $\delta_t$, SARSA on-policy vs Q-Learning off-policy, Q-table updates | `TDErrorVisualizer`, `QTableHeatmap`, `QlearningExplainer` |
| **05. Policy Gradients & Actor-Critic** | 📋 **Planned (v0.5.0)** | REINFORCE, Policy parameterization $\pi_\theta$, Baseline subtraction, Advantage function | `PolicyLandscapeScene`, `ActorCriticLoop`, `PolicyGradientExplainer` |

---

## 5. Acceptance Criteria & Definition of Done

A module is considered **Done** only when:
1. **Simulation Correctness:** Pure Python simulation generates verifiable numbers matching canonical theoretical regret/value curves.
2. **Deterministic Reproducibility:** Random seeds are fixed; rerun produces identical frame-by-frame values.
3. **Typography & Styling:** No text clipping or overlapping; LaTeX compiles clean with `basic` template; font renders via `Helvetica`.
4. **All Three Scenes Render:** Both component scenes and the composite explainer execute without error.
5. **Video Artifact Export:** Production MP4 files exist in `./videos/` at 1080p60 quality.
6. **Documentation Complete:** Module `README.md` includes complete mathematical formulas, practical trade-offs, and step-by-step CLI execution instructions.
