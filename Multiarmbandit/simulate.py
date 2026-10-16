#!/usr/bin/env python3
"""
Multi-Armed Bandit Terminal Simulator
Pure Python implementation (zero graphics / headless) for fast terminal execution.

Models the 3-headline conversion experiment from Sutton & Barto / RL-BLOG-ANIMATION:
  - Arm 0: "Save 20%"    (True mean: 0.20)
  - Arm 1: "Free shipping"(True mean: 0.45)
  - Arm 2: "Members save" (True mean: 0.70) [Optimal Arm]

Supported Algorithms:
  1. Epsilon-Greedy (classic exploration vs exploitation)
  2. Upper Confidence Bound 1 (UCB1 - optimism in the face of uncertainty)
  3. Thompson Sampling (Bayesian posterior Beta distribution sampling)
  4. Softmax / Boltzmann (temperature-scaled probability distribution)

Usage:
  python3 simulate.py                           # Runs comparison of all 4 algorithms
  python3 simulate.py --algo epsilon-greedy -v  # Step-by-step 36-round verbose trace
  python3 simulate.py --algo ucb --rounds 1000  # Long-horizon benchmark
  python3 simulate.py --algo thompson -v
"""

import argparse
import math
import random
from typing import List, Tuple, Dict


# --- Environment ---

class BernoulliBandit:
    """Multi-armed bandit environment with Bernoulli reward distributions."""

    def __init__(self, arm_names: List[str], true_means: List[float]):
        assert len(arm_names) == len(true_means), "Names and means must have equal length"
        self.arm_names = arm_names
        self.true_means = true_means
        self.k = len(true_means)
        self.optimal_mean = max(true_means)
        self.optimal_arm = true_means.index(self.optimal_mean)

    def pull(self, arm_idx: int) -> Tuple[int, float]:
        """
        Pull arm `arm_idx`.
        Returns:
            reward: 1 (success/click) or 0 (failure/miss)
            instantaneous_regret: optimal_mean - true_mean[arm_idx]
        """
        success_prob = self.true_means[arm_idx]
        reward = 1 if random.random() < success_prob else 0
        instantaneous_regret = self.optimal_mean - success_prob
        return reward, instantaneous_regret


# --- Policies ---

class EpsilonGreedyPolicy:
    """Epsilon-Greedy: explores randomly with probability epsilon, exploits best mean otherwise."""

    def __init__(self, k: int, epsilon: float = 0.15):
        self.k = k
        self.epsilon = epsilon
        self.name = f"Epsilon-Greedy (ε={epsilon})"
        self.counts = [0] * k
        self.estimates = [0.0] * k

    def select_arm(self, round_num: int) -> Tuple[int, str]:
        # Pull each arm at least once initially
        for arm_idx in range(self.k):
            if self.counts[arm_idx] == 0:
                return arm_idx, "explore"

        if random.random() < self.epsilon:
            return random.randrange(self.k), "explore"
        else:
            max_val = max(self.estimates)
            # Break ties uniformly at random
            candidates = [i for i, val in enumerate(self.estimates) if val == max_val]
            return random.choice(candidates), "exploit"

    def update(self, arm_idx: int, reward: int):
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        # Incremental running mean update: Q_n = Q_{n-1} + (R - Q_{n-1}) / n
        self.estimates[arm_idx] += (reward - self.estimates[arm_idx]) / n


class UCB1Policy:
    """Upper Confidence Bound (UCB1): selects arm maximizing Q_i + c * sqrt(ln(t) / N_i)."""

    def __init__(self, k: int, c: float = 1.0):
        self.k = k
        self.c = c
        self.name = f"UCB1 (c={c})"
        self.counts = [0] * k
        self.estimates = [0.0] * k

    def select_arm(self, round_num: int) -> Tuple[int, str]:
        # Pull each arm once to initialize counts
        for arm_idx in range(self.k):
            if self.counts[arm_idx] == 0:
                return arm_idx, "explore"

        t = round_num + 1
        ucb_values = []
        for i in range(self.k):
            bonus = self.c * math.sqrt(math.log(t) / self.counts[i])
            ucb_values.append(self.estimates[i] + bonus)

        max_val = max(ucb_values)
        candidates = [i for i, val in enumerate(ucb_values) if math.isclose(val, max_val)]
        chosen = random.choice(candidates)
        # Check if chosen arm has highest raw estimate (exploit) or pulled due to bonus (explore)
        mode = "exploit" if math.isclose(self.estimates[chosen], max(self.estimates)) else "explore"
        return chosen, mode

    def update(self, arm_idx: int, reward: int):
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.estimates[arm_idx] += (reward - self.estimates[arm_idx]) / n


class ThompsonSamplingPolicy:
    """Thompson Sampling: samples from Beta(alpha_i, beta_i) posteriors for Bernoulli rewards."""

    def __init__(self, k: int):
        self.k = k
        self.name = "Thompson Sampling (Beta Prior)"
        # Prior Beta(1, 1) represents uniform prior on [0, 1]
        self.alphas = [1] * k
        self.betas = [1] * k
        self.counts = [0] * k

    @property
    def estimates(self) -> List[float]:
        # Posterior mean of Beta(alpha, beta) is alpha / (alpha + beta)
        return [self.alphas[i] / (self.alphas[i] + self.betas[i]) for i in range(self.k)]

    def select_arm(self, round_num: int) -> Tuple[int, str]:
        samples = [random.betavariate(self.alphas[i], self.betas[i]) for i in range(self.k)]
        max_sample = max(samples)
        candidates = [i for i, s in enumerate(samples) if s == max_sample]
        chosen = random.choice(candidates)

        # Classify as exploit if it matches the current posterior mean leader
        best_mean = max(self.estimates)
        mode = "exploit" if math.isclose(self.estimates[chosen], best_mean) else "explore"
        return chosen, mode

    def update(self, arm_idx: int, reward: int):
        self.counts[arm_idx] += 1
        if reward == 1:
            self.alphas[arm_idx] += 1
        else:
            self.betas[arm_idx] += 1


class SoftmaxPolicy:
    """Softmax (Boltzmann): selects arms according to Gibbs/Boltzmann distribution over estimates."""

    def __init__(self, k: int, temperature: float = 0.2):
        self.k = k
        self.temperature = max(1e-4, temperature)
        self.name = f"Softmax (τ={temperature})"
        self.counts = [0] * k
        self.estimates = [0.0] * k

    def select_arm(self, round_num: int) -> Tuple[int, str]:
        for arm_idx in range(self.k):
            if self.counts[arm_idx] == 0:
                return arm_idx, "explore"

        # Numerical stability trick: subtract max estimate before exponentiation
        max_est = max(self.estimates)
        exp_vals = [math.exp((q - max_est) / self.temperature) for q in self.estimates]
        total_exp = sum(exp_vals)
        probs = [v / total_exp for v in exp_vals]

        r = random.random()
        cumulative = 0.0
        chosen = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen = i
                break

        mode = "exploit" if math.isclose(self.estimates[chosen], max_est) else "explore"
        return chosen, mode

    def update(self, arm_idx: int, reward: int):
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.estimates[arm_idx] += (reward - self.estimates[arm_idx]) / n


class DecayingEpsilonGreedyPolicy:
    """Decaying Epsilon-Greedy: epsilon decays with time t: eps_t = min(1.0, c * K / t)."""

    def __init__(self, k: int, c: float = 1.0):
        self.k = k
        self.c = c
        self.name = f"Decaying ε-Greedy (c={c})"
        self.counts = [0] * k
        self.estimates = [0.0] * k

    def select_arm(self, round_num: int) -> Tuple[int, str]:
        for arm_idx in range(self.k):
            if self.counts[arm_idx] == 0:
                return arm_idx, "explore"

        t = round_num + 1
        eps_t = min(1.0, (self.c * self.k) / t)

        if random.random() < eps_t:
            return random.randrange(self.k), "explore"
        else:
            max_val = max(self.estimates)
            candidates = [i for i, val in enumerate(self.estimates) if val == max_val]
            return random.choice(candidates), "exploit"

    def update(self, arm_idx: int, reward: int):
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.estimates[arm_idx] += (reward - self.estimates[arm_idx]) / n


class OptimisticInitialValuesPolicy:
    """Optimistic Initial Values: sets high initial Q_0 (e.g. 1.5) and acts purely greedy."""

    def __init__(self, k: int, initial_value: float = 1.5):
        self.k = k
        self.initial_value = initial_value
        self.name = f"Optimistic Initial (Q₀={initial_value})"
        self.counts = [0] * k
        self.estimates = [initial_value] * k

    def select_arm(self, round_num: int) -> Tuple[int, str]:
        max_val = max(self.estimates)
        candidates = [i for i, val in enumerate(self.estimates) if math.isclose(val, max_val)]
        chosen = random.choice(candidates)
        mode = "explore" if any(c == 0 for c in self.counts) else "exploit"
        return chosen, mode

    def update(self, arm_idx: int, reward: int):
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.estimates[arm_idx] += (reward - self.estimates[arm_idx]) / n


class GradientBanditPolicy:
    """Gradient Bandit: learns preference H_t(a) via policy gradient with average reward baseline."""

    def __init__(self, k: int, alpha: float = 0.1):
        self.k = k
        self.alpha = alpha
        self.name = f"Gradient Bandit (α={alpha})"
        self.preferences = [0.0] * k
        self.counts = [0] * k
        self.total_reward = 0
        self.total_pulls = 0
        self.probabilities = [1.0 / k] * k
        self._empirical_estimates = [0.0] * k

    @property
    def estimates(self) -> List[float]:
        return self._empirical_estimates

    def _compute_probabilities(self) -> List[float]:
        max_pref = max(self.preferences)
        exp_vals = [math.exp(p - max_pref) for p in self.preferences]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]

    def select_arm(self, round_num: int) -> Tuple[int, str]:
        self.probabilities = self._compute_probabilities()
        r = random.random()
        cumulative = 0.0
        chosen = 0
        for i, p in enumerate(self.probabilities):
            cumulative += p
            if r <= cumulative:
                chosen = i
                break

        max_prob = max(self.probabilities)
        mode = "exploit" if math.isclose(self.probabilities[chosen], max_prob) else "explore"
        return chosen, mode

    def update(self, arm_idx: int, reward: int):
        self.counts[arm_idx] += 1
        self.total_pulls += 1
        self.total_reward += reward
        baseline = self.total_reward / self.total_pulls

        n = self.counts[arm_idx]
        self._empirical_estimates[arm_idx] += (reward - self._empirical_estimates[arm_idx]) / n

        for i in range(self.k):
            if i == arm_idx:
                self.preferences[i] += self.alpha * (reward - baseline) * (1.0 - self.probabilities[i])
            else:
                self.preferences[i] -= self.alpha * (reward - baseline) * self.probabilities[i]


# --- Runner & Reporter ---

def run_simulation(env: BernoulliBandit, policy, rounds: int, verbose: bool = False) -> Dict:
    total_reward = 0
    cumulative_regret = 0.0
    history = []

    if verbose:
        print(f"\n{'='*75}")
        print(f" Running Simulation: {policy.name}")
        print(f" Horizon: {rounds} rounds | Arms: {env.k}")
        print(f"{'='*75}")
        print(f"{'Round':<6} | {'Chosen Arm':<15} | {'Mode':<8} | {'Outcome':<8} | {'Est. μ':<22} | {'Cum. Regret':<10}")
        print(f"{'-'*75}")

    for t in range(rounds):
        arm_idx, mode = policy.select_arm(t)
        reward, inst_regret = env.pull(arm_idx)
        policy.update(arm_idx, reward)

        total_reward += reward
        cumulative_regret += inst_regret

        history.append({
            "round": t + 1,
            "arm": arm_idx,
            "arm_name": env.arm_names[arm_idx],
            "mode": mode,
            "reward": reward,
            "cum_regret": cumulative_regret,
        })

        if verbose:
            out_str = "✓ +1" if reward == 1 else "✗ +0"
            est_str = "[" + ", ".join(f"{q:.2f}" for q in policy.estimates) + "]"
            mode_tag = f"\033[33m{mode.upper()}\033[0m" if mode == "explore" else f"\033[32m{mode.upper()}\033[0m"
            print(f"{t+1:<6} | {env.arm_names[arm_idx]:<15} | {mode_tag:<17} | {out_str:<8} | {est_str:<22} | {cumulative_regret:<10.2f}")

    optimal_pulls = policy.counts[env.optimal_arm]
    optimal_rate = (optimal_pulls / rounds) * 100.0

    return {
        "name": policy.name,
        "rounds": rounds,
        "total_reward": total_reward,
        "cumulative_regret": cumulative_regret,
        "optimal_pulls": optimal_pulls,
        "optimal_rate": optimal_rate,
        "counts": policy.counts,
        "estimates": policy.estimates,
    }


def print_comparison_table(env: BernoulliBandit, results: List[Dict]):
    print("\n" + "=" * 94)
    print(f" MULTI-ARMED BANDIT BENCHMARK SUMMARY (Horizon: {results[0]['rounds']} rounds)")
    print("=" * 94)
    header = f"{'Algorithm':<32} | {'Reward':<8} | {'Opt. Arm %':<11} | {'Cum. Regret':<12} | {'Estimated vs True Click Rates'}"
    print(header)
    print("-" * 94)

    for res in results:
        est_strs = [f"{est:.2f} (true {env.true_means[i]:.2f})" for i, est in enumerate(res["estimates"])]
        est_summary = " | ".join(est_strs)
        print(f"{res['name']:<32} | {res['total_reward']:<8} | {res['optimal_rate']:>8.1f}%   | {res['cumulative_regret']:>11.2f}  | {est_summary}")

    print("-" * 94)
    print(f"* Optimal Arm: Arm {env.optimal_arm} ('{env.arm_names[env.optimal_arm]}', μ* = {env.optimal_mean:.2f})")
    print(f"* Theoretical minimum expected regret is achieved by picking Arm {env.optimal_arm} on every step.\n")


# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Headless Multi-Armed Bandit Terminal Simulator (7 Supported Policies)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        choices=["epsilon-greedy", "decaying-epsilon", "oiv", "ucb", "thompson", "softmax", "gradient", "all"],
        default="all",
        help="Bandit algorithm to simulate",
    )
    parser.add_argument("--rounds", "-t", type=int, default=36, help="Number of time steps/pulls")
    parser.add_argument("--epsilon", "-e", type=float, default=0.15, help="Epsilon for Epsilon-Greedy")
    parser.add_argument("--decay-c", type=float, default=1.0, help="Decay constant for Decaying Epsilon-Greedy")
    parser.add_argument("--q0", type=float, default=1.5, help="Initial value for Optimistic Initial Values")
    parser.add_argument("--c", type=float, default=1.0, help="Exploration constant for UCB1")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature for Softmax")
    parser.add_argument("--alpha", type=float, default=0.1, help="Step-size learning rate for Gradient Bandit")
    parser.add_argument("--seed", "-s", type=int, default=16, help="Random seed for reproducibility")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print step-by-step trace table")
    args = parser.parse_args()

    # Reproducibility seed (seed=16 reproduces the canonical explainer trace)
    random.seed(args.seed)

    headlines = ["Save 20%", "Free shipping", "Members save"]
    true_means = [0.20, 0.45, 0.70]
    env = BernoulliBandit(headlines, true_means)

    policies = {
        "epsilon-greedy": lambda: EpsilonGreedyPolicy(env.k, epsilon=args.epsilon),
        "decaying-epsilon": lambda: DecayingEpsilonGreedyPolicy(env.k, c=args.decay_c),
        "oiv": lambda: OptimisticInitialValuesPolicy(env.k, initial_value=args.q0),
        "ucb": lambda: UCB1Policy(env.k, c=args.c),
        "thompson": lambda: ThompsonSamplingPolicy(env.k),
        "softmax": lambda: SoftmaxPolicy(env.k, temperature=args.temp),
        "gradient": lambda: GradientBanditPolicy(env.k, alpha=args.alpha),
    }

    if args.algo == "all":
        results = []
        for name, factory in policies.items():
            # Reset seed per policy so they face identical underlying environments
            random.seed(args.seed)
            pol = factory()
            res = run_simulation(env, pol, args.rounds, verbose=args.verbose)
            results.append(res)
        print_comparison_table(env, results)
    else:
        pol = policies[args.algo]()
        res = run_simulation(env, pol, args.rounds, verbose=True)
        print_comparison_table(env, [res])


if __name__ == "__main__":
    main()
