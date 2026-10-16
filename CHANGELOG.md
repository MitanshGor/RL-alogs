# CHANGELOG.md

All notable changes to the `RL-BLOG-ANIMATION` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- **MDP Module (`02-MDP`):**
  - Markov Decision Process graph visualizer with transition probabilities $P(s' \mid s, a)$ and discount factor $\gamma$.
  - State value function $V(s)$ vs action value function $Q(s, a)$ visual breakdown.
- **Bellman Equation & Dynamic Programming (`03-Bellman-DP`):**
  - 4x4 Gridworld environment with policy iteration and value iteration sweeps.
  - Interactive backup tree animations.
- **Automated Web Preview:** Lightweight HTML5 video gallery embedding rendered `.mp4` artifacts.

---

## [0.1.0] - 2026-10-16

### Added
- **Multi-Armed Bandit Module (`Multiarmbandit/`):**
  - Mathematical formalization of the multi-armed bandit problem, instantaneous regret, and cumulative regret.
  - Headless terminal simulation runner (`simulate.py`) with zero graphic dependencies, implementing 7 policies (Epsilon-Greedy, Decaying-Epsilon, Optimistic Initial Values, UCB1, Thompson Sampling, Softmax, and Gradient Bandit).
  - Dynamic simulation trace generator (`generate_pull_trace`) bridging pure Python algorithmic policies into ManimGL vector animations.
  - Interactive 36-round Epsilon-Greedy simulation trace with fixed seed (`seed=16`).
  - Real-world headline testing case study with three variants ("Save 20%", "Free shipping", "Members save").
- **Empirical Visualizations Suite (`plot_comparisons.py` & `Multiarmbandit/visualizations/`):**
  - Generated 6 publication-grade 300 DPI research figures and a master composite 6-panel dashboard comparing all 7 policies:
    1. Cumulative Regret vs Time ($\mathcal{O}(t)$ linear vs $\mathcal{O}(\log t)$ logarithmic).
    2. % Optimal Arm Selection over time (Sutton & Barto Fig 2.2).
    3. Arm Allocation dynamic stacked area chart.
    4. Posterior Belief Evolution (Beta density curves for Thompson Sampling).
    5. Parameter Sensitivity Sweep (Sutton & Barto Fig 2.6 inverted-U curves).
    6. Monte Carlo Regret Distribution & Downside Risk boxplots (100 seeds).
- **ManimGL Animation Suite (`multiarmed_bandit.py`):**
  - `BanditDefinition`: Visual explanation of arms, unknown reward distributions $\mu_i$, and regret intuition.
  - `BanditUsageExample`: Real-time dashboard showing running estimate bars climbing toward hidden ground-truth marks, explore/exploit state badges, and cumulative regret counter.
  - `UCBUsageExample` & `ThompsonUsageExample`: Dynamically generated animation scenes for UCB and Thompson Sampling.
  - `BanditExplainer`: Composite master scene sequencing definition and usage trace.
- **Pre-Rendered Video Assets (`Multiarmbandit/videos/`):**
  - `BanditDefinition.mp4`
  - `BanditUsageExample.mp4`
  - `BanditExplainer.mp4`
- **Cross-Platform & macOS Configuration (`custom_config.yml`):**
  - Configured `basic` TeX template override to eliminate 3b1b package compilation crashes on standard LaTeX installs.
  - Configured `Helvetica` font fallback for clean typography on macOS.
- **Vibe-Coding Infrastructure & Documentation:**
  - `CLAUDE.md`: System instructions, build commands, and AI guardrails.
  - `.cursorrules`: Fast-context rules and execution commands for Cursor IDE.
  - `AGENTS.md`: Multi-agent behavioral roles, permission boundaries, and Test-Driven Animation (TDA) loops.
  - `PRD.md`: Product requirements, user stories, scope boundaries, and curriculum roadmap.
  - `TECH_SPEC.md`: Architecture diagrams, scene hierarchies, data schemas, and rendering flags.
  - `CONTRIBUTING.md`: Human-AI collaboration protocol, commit conventions, and branch strategy.
