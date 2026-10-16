#!/usr/bin/env python3
"""
Multi-Armed Bandit Comparison Visualizations Suite
Generates 6 canonical RL research plots comparing the 7 supported policies:

1. Cumulative Regret vs Time Horizon (O(t) linear vs O(log t) logarithmic)
2. % Optimal Arm Selection over Time (Sutton & Barto Fig 2.2)
3. Arm Allocation Stacked Area Chart (Dynamic portfolio shift over time)
4. Posterior Belief Evolution (Beta distributions for Thompson Sampling)
5. Parameter Sensitivity Sweep (Sutton & Barto Fig 2.6 inverted-U curves)
6. Monte Carlo Distribution & Risk Box/Violin Plots (100-seed variance)
+ Master Composite Dashboard (all 6 combined into a publication-grade graphic)

Outputs saved to: Multiarmbandit/visualizations/
"""

import os
import sys
import math
import random
from typing import List, Dict, Tuple

# Use Agg backend for headless generation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.ndimage import uniform_filter1d

# Import simulation environment and policies
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from simulate import (
    BernoulliBandit,
    EpsilonGreedyPolicy,
    DecayingEpsilonGreedyPolicy,
    OptimisticInitialValuesPolicy,
    UCB1Policy,
    ThompsonSamplingPolicy,
    SoftmaxPolicy,
    GradientBanditPolicy,
)

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetic Theme Tokens
PALETTE = {
    "Epsilon-Greedy": "#e11d48",       # Crimson Red
    "Decaying ε-Greedy": "#f97316",     # Orange
    "Optimistic Initial": "#eab308",    # Gold
    "UCB1": "#0d9488",                  # Teal
    "Thompson Sampling": "#2563eb",     # Royal Blue
    "Softmax": "#8b5cf6",               # Purple
    "Gradient Bandit": "#059669",       # Emerald Green
}

ARM_COLORS = ["#3b82f6", "#14b8a6", "#f59e0b"]  # Blue, Teal, Amber
ARM_NAMES = ["Save 20%", "Free shipping", "Members save"]
TRUE_MEANS = [0.20, 0.45, 0.70]
OPTIMAL_ARM = 2
OPTIMAL_MEAN = 0.70


def setup_matplotlib_style():
    """Configure modern publication styling."""
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",    # Dark slate background
        "axes.facecolor": "#1e293b",      # Slightly lighter card slate
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#f8fafc",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "text.color": "#f8fafc",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
        "font.size": 11,
    })


def get_policy_factories():
    return {
        "Epsilon-Greedy": lambda k: EpsilonGreedyPolicy(k, epsilon=0.15),
        "Decaying ε-Greedy": lambda k: DecayingEpsilonGreedyPolicy(k, c=1.0),
        "Optimistic Initial": lambda k: OptimisticInitialValuesPolicy(k, initial_value=1.5),
        "UCB1": lambda k: UCB1Policy(k, c=1.0),
        "Thompson Sampling": lambda k: ThompsonSamplingPolicy(k),
        "Softmax": lambda k: SoftmaxPolicy(k, temperature=0.2),
        "Gradient Bandit": lambda k: GradientBanditPolicy(k, alpha=0.1),
    }


# ==============================================================================
# Simulation Multi-Run Engine
# ==============================================================================

def run_multi_seed_experiment(rounds=500, num_seeds=50) -> Dict:
    """Runs multiple seeds across all 7 policies to calculate statistical averages."""
    print(f"[*] Running Multi-Seed Experiment ({num_seeds} seeds x {rounds} rounds across 7 policies)...")
    env = BernoulliBandit(ARM_NAMES, TRUE_MEANS)
    factories = get_policy_factories()

    # Data structures: policy -> (seeds x rounds) arrays
    regret_curves = {name: np.zeros((num_seeds, rounds)) for name in factories}
    optimal_arm_pulls = {name: np.zeros((num_seeds, rounds)) for name in factories}
    rewards = {name: np.zeros((num_seeds, rounds)) for name in factories}
    total_regrets = {name: [] for name in factories}

    for seed in range(num_seeds):
        for name, factory in factories.items():
            random.seed(seed + 1000)
            pol = factory(env.k)
            cum_reg = 0.0

            for t in range(rounds):
                arm_idx, _ = pol.select_arm(t)
                reward, inst_reg = env.pull(arm_idx)
                pol.update(arm_idx, reward)

                cum_reg += inst_reg
                regret_curves[name][seed, t] = cum_reg
                optimal_arm_pulls[name][seed, t] = 1.0 if arm_idx == OPTIMAL_ARM else 0.0
                rewards[name][seed, t] = reward

            total_regrets[name].append(cum_reg)

    return {
        "regret_curves": regret_curves,
        "optimal_arm_pulls": optimal_arm_pulls,
        "rewards": rewards,
        "total_regrets": total_regrets,
        "rounds": rounds,
        "num_seeds": num_seeds,
    }


# ==============================================================================
# Plot 1: Cumulative Regret vs Time Horizon
# ==============================================================================

def plot_1_cumulative_regret(data: Dict, ax=None, standalone=True):
    print("  -> Generating Plot 1: Cumulative Regret vs Time Horizon...")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    else:
        fig = None

    rounds = data["rounds"]
    x = np.arange(1, rounds + 1)

    for name, matrix in data["regret_curves"].items():
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        sem = std / np.sqrt(data["num_seeds"])

        color = PALETTE[name]
        ax.plot(x, mean, label=name, color=color, linewidth=2.2)
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.15)

    ax.set_title("1. Cumulative Regret vs. Time Horizon (Lower is Better)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rounds (t)", fontsize=11)
    ax.set_ylabel(r"Cumulative Regret $\mathcal{R}_t$", fontsize=11)
    ax.grid(True)
    ax.legend(loc="upper left", framealpha=0.8, fontsize=9.5)

    # Annotation highlighting theoretical separation
    ax.annotate(r"Linear Regret $\mathcal{O}(t)$" + "\n(Static Exploration)", 
                xy=(rounds * 0.78, np.mean(data["regret_curves"]["Epsilon-Greedy"], axis=0)[int(rounds * 0.78)]),
                xytext=(rounds * 0.55, np.mean(data["regret_curves"]["Epsilon-Greedy"], axis=0)[-1] * 0.85),
                arrowprops=dict(arrowstyle="->", color="#f43f5e", lw=1.5),
                color="#f43f5e", fontsize=9, fontweight="bold")

    ax.annotate(r"Logarithmic Regret $\mathcal{O}(\log t)$" + "\n(Optimal Convergence)", 
                xy=(rounds * 0.85, np.mean(data["regret_curves"]["Thompson Sampling"], axis=0)[int(rounds * 0.85)]),
                xytext=(rounds * 0.55, 12),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.5),
                color="#38bdf8", fontsize=9, fontweight="bold")

    if standalone:
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "01_cumulative_regret.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"     Saved: {out_path}")


# ==============================================================================
# Plot 2: % Optimal Arm Selection over Time
# ==============================================================================

def plot_2_optimal_arm_percentage(data: Dict, ax=None, standalone=True):
    print("  -> Generating Plot 2: % Optimal Arm Selection over Time...")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    else:
        fig = None

    rounds = data["rounds"]
    x = np.arange(1, rounds + 1)
    window = 15  # Smoothing window

    for name, matrix in data["optimal_arm_pulls"].items():
        # Average across seeds, then compute smooth rolling mean
        raw_prob = np.mean(matrix, axis=0) * 100.0
        smoothed = uniform_filter1d(raw_prob, size=window, mode='nearest')

        color = PALETTE[name]
        ax.plot(x, smoothed, label=name, color=color, linewidth=2.0)

    # Theoretical ceiling for Epsilon-Greedy (1 - eps + eps/K = 1 - 0.15 + 0.05 = 90%)
    ax.axhline(90.0, color="#f43f5e", linestyle=":", linewidth=1.8, label=r"$\epsilon$-Greedy Theoretical Cap (90%)")
    # Random guessing baseline (1/3 = 33.3%)
    ax.axhline(33.33, color="#94a3b8", linestyle="--", linewidth=1.2, label="Random Baseline (33.3%)")

    ax.set_title("2. % Optimal Arm Selection Over Time (Higher is Better)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rounds (t)", fontsize=11)
    ax.set_ylabel("% Optimal Arm Pulls", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.8, fontsize=9)

    if standalone:
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "02_optimal_arm_percentage.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"     Saved: {out_path}")


# ==============================================================================
# Plot 3: Arm Allocation Stacked Area Chart (Dynamic Portfolio Shift)
# ==============================================================================

def plot_3_arm_allocation_stream(rounds=500, num_seeds=40, ax=None, standalone=True):
    print("  -> Generating Plot 3: Arm Allocation Stacked Area Chart...")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    else:
        fig = None

    env = BernoulliBandit(ARM_NAMES, TRUE_MEANS)
    # Track pull proportions over time for Thompson Sampling
    pull_history = np.zeros((3, rounds))

    for seed in range(num_seeds):
        random.seed(seed + 42)
        pol = ThompsonSamplingPolicy(env.k)
        for t in range(rounds):
            arm_idx, _ = pol.select_arm(t)
            reward, _ = env.pull(arm_idx)
            pol.update(arm_idx, reward)
            pull_history[arm_idx, t] += 1

    # Normalize to percentages
    total_pulls_per_t = np.sum(pull_history, axis=0)
    proportions = (pull_history / total_pulls_per_t) * 100.0

    # Smooth curves with uniform_filter1d to preserve edges
    window = 20
    smoothed = np.zeros_like(proportions)
    for i in range(3):
        smoothed[i] = uniform_filter1d(proportions[i], size=window, mode='nearest')
    # Renormalize to guarantee sum is exactly 100% at each time step
    smoothed = (smoothed / np.sum(smoothed, axis=0)) * 100.0

    x = np.arange(1, rounds + 1)
    ax.stackplot(
        x,
        smoothed[0], smoothed[1], smoothed[2],
        labels=[f"Arm 0: {ARM_NAMES[0]} (μ=0.20)",
                f"Arm 1: {ARM_NAMES[1]} (μ=0.45)",
                f"Arm 2: {ARM_NAMES[2]} (μ=0.70) [Optimal]"],
        colors=ARM_COLORS,
        alpha=0.85,
    )

    ax.set_title("3. Arm Allocation Shift Over Time (Thompson Sampling)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rounds (t)", fontsize=11)
    ax.set_ylabel("Share of Total Selections (%)", fontsize=11)
    ax.set_xlim(1, rounds)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", framealpha=0.9, fontsize=9.5)

    if standalone:
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "03_arm_allocation_stream.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"     Saved: {out_path}")


# ==============================================================================
# Plot 4: Posterior Belief Evolution (Beta Distributions)
# ==============================================================================

def plot_4_posterior_evolution(standalone=True):
    print("  -> Generating Plot 4: Posterior Belief Evolution (Beta Curves)...")
    env = BernoulliBandit(ARM_NAMES, TRUE_MEANS)
    random.seed(16)
    pol = ThompsonSamplingPolicy(env.k)

    snapshots = [0, 10, 50, 150, 500]
    snapshot_data = {}

    snapshot_data[0] = (list(pol.alphas), list(pol.betas))
    for t in range(1, 501):
        arm_idx, _ = pol.select_arm(t - 1)
        reward, _ = env.pull(arm_idx)
        pol.update(arm_idx, reward)
        if t in snapshots:
            snapshot_data[t] = (list(pol.alphas), list(pol.betas))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), dpi=300, sharey=True)
    theta = np.linspace(0.001, 0.999, 400)

    for idx, (t, (alphas, betas)) in enumerate(snapshot_data.items()):
        ax = axes[idx]
        for arm_i in range(3):
            a, b = alphas[arm_i], betas[arm_i]
            y = beta_dist.pdf(theta, a, b)
            ax.plot(theta, y, color=ARM_COLORS[arm_i], linewidth=2.2, label=ARM_NAMES[arm_i] if idx == 0 else "")
            # True mean vertical marker
            ax.axvline(TRUE_MEANS[arm_i], color=ARM_COLORS[arm_i], linestyle=":", alpha=0.7)

        title_str = f"Round t = {t}" if t > 0 else "Prior (t = 0)"
        ax.set_title(title_str, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Reward Probability θ", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Probability Density f(θ)", fontsize=11)
        ax.grid(True)
        ax.set_xlim(0, 1)

    fig.suptitle("4. Thompson Sampling Posterior Distribution Evolution (Uncertainty Collapse)", 
                 fontsize=14, fontweight="bold", y=1.03)
    axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.8)

    if standalone:
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "04_thompson_posterior_evolution.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"     Saved: {out_path}")


# ==============================================================================
# Plot 5: Parameter Sensitivity Sweep (Sutton & Barto Fig 2.6)
# ==============================================================================

def plot_5_parameter_sensitivity(ax=None, standalone=True):
    print("  -> Generating Plot 5: Parameter Sensitivity Sweep (Inverted-U Curves)...")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    else:
        fig = None

    env = BernoulliBandit(ARM_NAMES, TRUE_MEANS)
    rounds = 300
    seeds_per_param = 25

    # Parameter ranges across algorithms
    param_configs = {
        "Epsilon-Greedy (ε)": {
            "values": [0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6],
            "factory": lambda val: EpsilonGreedyPolicy(env.k, epsilon=val),
            "color": PALETTE["Epsilon-Greedy"],
        },
        "UCB1 (c)": {
            "values": [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
            "factory": lambda val: UCB1Policy(env.k, c=val),
            "color": PALETTE["UCB1"],
        },
        "Optimistic Initial (Q₀)": {
            "values": [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0],
            "factory": lambda val: OptimisticInitialValuesPolicy(env.k, initial_value=val),
            "color": PALETTE["Optimistic Initial"],
        },
        "Softmax (τ)": {
            "values": [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2],
            "factory": lambda val: SoftmaxPolicy(env.k, temperature=val),
            "color": PALETTE["Softmax"],
        },
        "Gradient Bandit (α)": {
            "values": [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8],
            "factory": lambda val: GradientBanditPolicy(env.k, alpha=val),
            "color": PALETTE["Gradient Bandit"],
        },
    }

    for algo_name, config in param_configs.items():
        vals = config["values"]
        mean_rewards = []

        for p_val in vals:
            rewards_list = []
            for s in range(seeds_per_param):
                random.seed(s + 500)
                pol = config["factory"](p_val)
                tot_rew = 0
                for t in range(rounds):
                    arm_idx, _ = pol.select_arm(t)
                    reward, _ = env.pull(arm_idx)
                    pol.update(arm_idx, reward)
                    tot_rew += reward
                rewards_list.append(tot_rew / rounds)
            mean_rewards.append(np.mean(rewards_list))

        ax.plot(vals, mean_rewards, marker="o", linewidth=2.2, label=algo_name, color=config["color"])

    ax.set_xscale("log")
    ax.set_title("5. Parameter Sensitivity Sweep (Sutton & Barto Fig 2.6)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Parameter Value (Log Scale: ε, c, Q₀, τ, α)", fontsize=11)
    ax.set_ylabel("Average Reward / Round", fontsize=11)
    ax.set_ylim(0.25, 0.72)
    # Optimal arm ceiling reference line
    ax.axhline(OPTIMAL_MEAN, color="#f59e0b", linestyle=":", label="Optimal Mean (μ* = 0.70)")
    ax.grid(True, which="both")
    ax.legend(loc="lower left", framealpha=0.85, fontsize=9)

    if standalone:
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "05_parameter_sensitivity.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"     Saved: {out_path}")


# ==============================================================================
# Plot 6: Monte Carlo Distribution / Risk Box & Violin Plots
# ==============================================================================

def plot_6_risk_distribution(data: Dict, ax=None, standalone=True):
    print("  -> Generating Plot 6: Monte Carlo Distribution & Risk Box Plots...")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    else:
        fig = None

    names = list(data["total_regrets"].keys())
    regret_lists = [data["total_regrets"][name] for name in names]
    colors = [PALETTE[name] for name in names]

    # Create horizontal boxplot sorted by median performance
    medians = [np.median(lst) for lst in regret_lists]
    sorted_indices = np.argsort(medians)  # Lowest regret (best) at top

    sorted_names = [names[i] for i in sorted_indices]
    sorted_regrets = [regret_lists[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    bp = ax.boxplot(
        sorted_regrets,
        vert=False,
        patch_artist=True,
        tick_labels=sorted_names,
        widths=0.6,
        medianprops=dict(color="#ffffff", linewidth=2.5),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(color="#94a3b8", linewidth=1.5),
        capprops=dict(color="#94a3b8", linewidth=1.5),
        flierprops=dict(marker="o", color="#f43f5e", alpha=0.6, markersize=4),
    )

    for patch, col in zip(bp['boxes'], sorted_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#ffffff")

    ax.set_title("6. Final Cumulative Regret Distribution (100 Seeds, T=500)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Final Cumulative Regret (Lower & Narrower is Better)", fontsize=11)
    ax.grid(True, axis="x")

    if standalone:
        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "06_monte_carlo_risk_distribution.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"     Saved: {out_path}")


# ==============================================================================
# Master Composite Dashboard (All 6 plots combined into a publication grid)
# ==============================================================================

def generate_master_dashboard(data: Dict):
    print("  -> Assembling Master Composite Dashboard (6-Panel Grid)...")
    fig = plt.figure(figsize=(24, 15), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4_placeholder = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    plot_1_cumulative_regret(data, ax=ax1, standalone=False)
    plot_2_optimal_arm_percentage(data, ax=ax2, standalone=False)
    plot_3_arm_allocation_stream(rounds=data["rounds"], num_seeds=30, ax=ax3, standalone=False)
    
    # Subplot 4 in grid: Simplified 3-snapshot view of Posterior Evolution
    env = BernoulliBandit(ARM_NAMES, TRUE_MEANS)
    random.seed(16)
    pol = ThompsonSamplingPolicy(env.k)
    for t in range(500):
        a_i, _ = pol.select_arm(t)
        r_i, _ = env.pull(a_i)
        pol.update(a_i, r_i)
    
    theta = np.linspace(0.001, 0.999, 300)
    for i in range(3):
        pdf = beta_dist.pdf(theta, pol.alphas[i], pol.betas[i])
        ax4_placeholder.plot(theta, pdf, color=ARM_COLORS[i], linewidth=2.4, label=f"{ARM_NAMES[i]} (μ={TRUE_MEANS[i]})")
        ax4_placeholder.axvline(TRUE_MEANS[i], color=ARM_COLORS[i], linestyle=":")
    ax4_placeholder.set_title("4. Thompson Posterior Beliefs at t=500", fontsize=13, fontweight="bold", pad=12)
    ax4_placeholder.set_xlabel("Reward Probability θ", fontsize=11)
    ax4_placeholder.set_ylabel("Probability Density f(θ)", fontsize=11)
    ax4_placeholder.grid(True)
    ax4_placeholder.legend(loc="upper right", fontsize=9, framealpha=0.8)

    plot_5_parameter_sensitivity(ax=ax5, standalone=False)
    plot_6_risk_distribution(data, ax=ax6, standalone=False)

    fig.suptitle(
        "MULTI-ARMED BANDIT RESEARCH COMPARISON: 7 POLICIES BENCHMARKED\n"
        "Landing Page Headline Conversion Model (Arms: 0.20, 0.45, 0.70 | Optimal Arm = 'Members save')",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    out_path = os.path.join(OUTPUT_DIR, "all_comparisons_dashboard.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Successfully generated Master Dashboard: {out_path}")


# ==============================================================================
# Main Runner
# ==============================================================================

def main():
    setup_matplotlib_style()
    print("=" * 70)
    print(" Generating Multi-Armed Bandit Visualization Benchmark Suite")
    print("=" * 70)

    # 1. Run multi-seed experiment for statistical plots (1, 2, 6)
    exp_data = run_multi_seed_experiment(rounds=500, num_seeds=50)

    # 2. Generate individual standalone plots
    plot_1_cumulative_regret(exp_data, standalone=True)
    plot_2_optimal_arm_percentage(exp_data, standalone=True)
    plot_3_arm_allocation_stream(rounds=500, num_seeds=40, standalone=True)
    plot_4_posterior_evolution(standalone=True)
    plot_5_parameter_sensitivity(standalone=True)
    plot_6_risk_distribution(exp_data, standalone=True)

    # 3. Generate master composite dashboard
    generate_master_dashboard(exp_data)

    print("=" * 70)
    print(f"All visualizations successfully saved in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
