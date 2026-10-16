# RL-BLOG-ANIMATION

> **Visual, mathematical, and algorithmic animations for Reinforcement Learning.**  
> Crafted with [3Blue1Brown's ManimGL (v1.7.2)](https://github.com/3b1b/manim), Python 3.11, LaTeX, and NumPy.

---

## 📚 Specialized Project Guidelines & Specs

This project is configured for **high-velocity AI-assisted vibe coding** with strict architectural and mathematical guardrails:

| Document | Purpose |
| :--- | :--- |
| [**`CLAUDE.md`**](CLAUDE.md) | Core system instructions, tech stack, build commands, and coding guardrails |
| [**`.cursorrules`**](.cursorrules) | Fast-context directives, shortcut flags, and rules for Cursor IDE |
| [**`AGENTS.md`**](AGENTS.md) | Multi-agent personas, permission boundaries, and Test-Driven Animation (TDA) loops |
| [**`PRD.md`**](PRD.md) | Product vision, user stories, curriculum roadmap, and feature boundaries |
| [**`TECH_SPEC.md`**](TECH_SPEC.md) | System architecture diagrams, data schemas, and rendering pipeline specs |
| [**`CHANGELOG.md`**](CHANGELOG.md) | Version history, shipped modules, and planned releases |
| [**`CONTRIBUTING.md`**](CONTRIBUTING.md) | Human-AI collaboration protocol, commit conventions, and branch strategy |

---

## 🚀 Quick Start

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Enter Topic Directory
```bash
cd Multiarmbandit
```

### 3. Run or Preview Animations
```bash
# Interactive Real-Time OpenGL Preview (press 'q' or 'Esc' to exit)
manimgl multiarmed_bandit.py BanditDefinition

# Fast Snapshot of the Final Frame (saves image without full render)
manimgl multiarmed_bandit.py BanditDefinition -s

# Low-Resolution Fast Preview (480p)
manimgl multiarmed_bandit.py BanditDefinition -l

# Full Production 1080p60 Render (saves to ./videos/)
manimgl multiarmed_bandit.py BanditExplainer -w
```

---

## 🗺️ Curriculum Modules

- [x] [**Module 01: Multi-Armed Bandit**](Multiarmbandit/README.md)
  - Mathematical definition: $\mu_i$, cumulative regret $\mathcal{R}_T$, exploration vs exploitation.
  - Algorithms: Epsilon-Greedy, Upper Confidence Bound (UCB), Thompson Sampling, Softmax.
  - 36-round real-world headline conversion simulation trace.
  - Pre-rendered videos in `Multiarmbandit/videos/`.
- [ ] **Module 02: Markov Decision Processes (MDP)** *(Planned)*
- [ ] **Module 03: Bellman Equations & Dynamic Programming** *(Planned)*
- [ ] **Module 04: Model-Free TD & Q-Learning** *(Planned)*
- [ ] **Module 05: Policy Gradients & Actor-Critic** *(Planned)*
