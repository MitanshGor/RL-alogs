# Multi-Armed Bandit: Problem Definition & Simulation Guide

> 📖 **Full Explainer:** For theoretical background, algorithm trade-offs, and Sutton & Barto context, see [**`README.md`**](README.md).

## The Problem Setting

You are optimizing click-through rates across three candidate headlines for a landing page:
- **Arm 0:** `"Save 20%"` $\to$ True click rate $\mu_0 = 0.20$
- **Arm 1:** `"Free shipping"` $\to$ True click rate $\mu_1 = 0.45$
- **Arm 2:** `"Members save"` $\to$ True click rate $\mu_2 = 0.70$ *(Optimal Arm)*

### The Dilemma
The agent receives only a binary reward $R_t \in \{0, 1\}$ (miss or click) after selecting an arm. It never sees the true probabilities $\mu_i$.

### Regret Objective
Maximize cumulative reward or minimize cumulative regret:
$$
\mathcal{R}_T = \sum_{t=1}^T (\mu^* - \mu_{A_t})
$$
where $\mu^* = 0.70$.

---

## Terminal Simulation (Headless)

Run the pure-Python simulation without any graphics or animation dependencies:

```bash
# 1. Compare all four algorithms side-by-side
python3 simulate.py

# 2. View step-by-step trace of Epsilon-Greedy over 36 rounds
python3 simulate.py --algo epsilon-greedy -v

# 3. Benchmark over 1,000 rounds
python3 simulate.py --rounds 1000

# 4. Test Thompson Sampling with custom seed
python3 simulate.py --algo thompson --seed 42 -v
```

### Supported Policies (7 Algorithms)
1. **Epsilon-Greedy (`--algo epsilon-greedy --epsilon 0.15`)**: Balances random exploration ($\epsilon$) with greedy exploitation ($1-\epsilon$).
2. **Decaying Epsilon-Greedy (`--algo decaying-epsilon --decay-c 1.0`)**: Exploits decay schedule $\epsilon_t = \min(1, \frac{c K}{t})$ to achieve asymptotic logarithmic regret.
3. **Optimistic Initial Values (`--algo oiv --q0 1.5`)**: Sets high initial reward priors ($Q_0$), forcing natural disappointment-driven early exploration.
4. **UCB1 (`--algo ucb --c 1.0`)**: Implements optimism in the face of uncertainty using upper confidence bounds.
5. **Thompson Sampling (`--algo thompson`)**: Bayesian posterior sampling with Beta conjugate priors.
6. **Softmax / Boltzmann (`--algo softmax --temp 0.2`)**: Probabilistic Gibbs/Boltzmann distribution weighted by empirical means.
7. **Gradient Bandit (`--algo gradient --alpha 0.1`)**: Softmax policy gradient learning relative action preferences $H_t(a)$ against an average reward baseline.
