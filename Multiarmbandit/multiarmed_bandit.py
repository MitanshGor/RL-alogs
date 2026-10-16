"""
Multi-armed bandit, drawn with 3Blue1Brown's Manim (ManimGL).

Renders two lessons, or the full explainer:

    manimgl multiarmed_bandit.py BanditDefinition
    manimgl multiarmed_bandit.py BanditUsageExample
    manimgl multiarmed_bandit.py BanditExplainer -w

BanditDefinition states the model on three arms and one observed reward.
BanditUsageExample runs an epsilon-greedy trace on three headlines.
BanditExplainer plays both, in that order.
"""

from manimlib import *
from manimlib.scene.scene import EndScene
import numpy as np
import os


ARM_COLORS = [BLUE_C, TEAL_C, GOLD_C]
HEADLINES = ("Save 20%", "Free shipping", "Members save")
# Hidden click rates. The agent never receives these numbers, only 0/1 clicks.
TRUE_MEANS = (0.20, 0.45, 0.70)
# arm index, click (1) or not (0), and whether this round explores or exploits.
# Thirty-six rounds of epsilon-greedy (seed 16). Early estimates swing between
# the arms; later rounds settle near the true click rates.
CANONICAL_PULLS = (
    (0, 0, "explore"),
    (1, 1, "explore"),
    (2, 1, "explore"),
    (1, 0, "exploit"),
    (0, 0, "explore"),
    (1, 0, "explore"),
    (1, 1, "explore"),
    (2, 0, "exploit"),
    (1, 0, "exploit"),
    (1, 1, "explore"),
    (0, 0, "explore"),
    (1, 1, "exploit"),
    (2, 1, "explore"),
    (2, 1, "exploit"),
    (2, 1, "exploit"),
    (0, 0, "explore"),
    (2, 1, "exploit"),
    (2, 0, "exploit"),
    (2, 0, "exploit"),
    (2, 1, "exploit"),
    (1, 0, "explore"),
    (2, 1, "exploit"),
    (0, 0, "explore"),
    (1, 0, "explore"),
    (2, 1, "exploit"),
    (2, 0, "exploit"),
    (2, 1, "exploit"),
    (2, 1, "exploit"),
    (0, 1, "explore"),
    (2, 1, "exploit"),
    (2, 0, "exploit"),
    (2, 1, "exploit"),
    (0, 0, "explore"),
    (2, 1, "exploit"),
    (0, 0, "explore"),
    (2, 1, "exploit"),
)

PULLS = CANONICAL_PULLS


def generate_pull_trace(policy_name="epsilon-greedy", seed=16, rounds=36, **kwargs):
    """
    Dynamically generates a sequence of (arm_index, reward, mode) pulls
    using the pure Python simulation engine in simulate.py.
    """
    try:
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
        import random
        random.seed(seed)
        env = BernoulliBandit(list(HEADLINES), list(TRUE_MEANS))
        
        policy_map = {
            "epsilon-greedy": lambda: EpsilonGreedyPolicy(env.k, epsilon=kwargs.get("epsilon", 0.15)),
            "decaying-epsilon": lambda: DecayingEpsilonGreedyPolicy(env.k, c=kwargs.get("c", 1.0)),
            "oiv": lambda: OptimisticInitialValuesPolicy(env.k, initial_value=kwargs.get("q0", 1.5)),
            "ucb": lambda: UCB1Policy(env.k, c=kwargs.get("c", 1.0)),
            "thompson": lambda: ThompsonSamplingPolicy(env.k),
            "softmax": lambda: SoftmaxPolicy(env.k, temperature=kwargs.get("temp", 0.2)),
            "gradient": lambda: GradientBanditPolicy(env.k, alpha=kwargs.get("alpha", 0.1)),
        }
        
        factory = policy_map.get(policy_name, policy_map["epsilon-greedy"])
        policy = factory()
        trace = []
        for t in range(rounds):
            arm_idx, mode = policy.select_arm(t)
            reward, _ = env.pull(arm_idx)
            policy.update(arm_idx, reward)
            trace.append((arm_idx, reward, mode))
        return tuple(trace)
    except Exception:
        return CANONICAL_PULLS


def expected_regret(pulls=None, means=TRUE_MEANS):
    if pulls is None:
        pulls = PULLS
    best = max(means)
    return sum(best - means[arm] for arm, _reward, _mode in pulls)


def make_machine(index, color):
    """One arm: a window for the reward, and a button the agent presses."""
    body = RoundedRectangle(width=2.45, height=2.85, corner_radius=0.14)
    body.set_stroke(color, width=2.5)
    body.set_fill(color, opacity=0.08)

    window = RoundedRectangle(width=1.7, height=1.05, corner_radius=0.08)
    window.set_fill(BLACK, 1)
    window.set_stroke(color, width=2)
    window.move_to(body.get_center() + 0.42 * UP)

    readout = Text("?", font_size=52)
    readout.move_to(window)
    reward = Text("1", font_size=52)
    reward.set_color(GREEN)
    reward.move_to(window)
    reward.set_opacity(0)

    name = Text(f"Arm {index}", font_size=28)
    name.set_color(color)
    name.next_to(window, DOWN, buff=0.22)

    button = Circle(radius=0.14)
    button.set_fill(color, opacity=1)
    button.set_stroke(WHITE, width=1)
    button.next_to(name, DOWN, buff=0.18)

    machine = VGroup(body, window, readout, reward, name, button)
    machine.body = body
    machine.window = window
    machine.readout = readout
    machine.reward = reward
    machine.button = button
    machine.arm_color = color
    return machine


def make_card(title, color):
    body = RoundedRectangle(width=3.55, height=1.42, corner_radius=0.12)
    body.set_stroke(color, width=2.5)
    body.set_fill(color, opacity=0.10)

    name = Text(title, font_size=30)
    name.move_to(body.get_center() + 0.24 * UP)

    blank = Text("-", font_size=26)
    blank.set_color(GREY_A)
    blank.next_to(name, DOWN, buff=0.14)
    click = Text("click", font_size=26)
    click.set_color(GREEN)
    click.move_to(blank)
    click.set_opacity(0)
    miss = Text("no click", font_size=26)
    miss.set_color(GREY_A)
    miss.move_to(blank)
    miss.set_opacity(0)

    card = VGroup(body, name, blank, click, miss)
    card.body = body
    card.name = name
    card.blank = blank
    card.click = click
    card.miss = miss
    card.arm_color = color
    return card


def fitted_text(words, font_size=30, max_width=12.6):
    mob = Text(words, font_size=font_size)
    if mob.get_width() > max_width:
        mob.set_width(max_width)
    return mob


class BanditScenes(Scene):
    def finish_batch_render(self):
        # ManimGL otherwise leaves the preview window open after -w.
        if os.environ.get("MANIMGL_EXIT_AFTER"):
            raise EndScene()

    def scoreboard(self, t, mode, reward, total_reward, regret_added, total_regret):
        """One round's reward beside the regret that same round added."""
        mode_color = ORANGE if mode == "explore" else GREEN if mode == "exploit" else WHITE
        if reward is None:
            top = fitted_text(f"t = {t}    {mode}", font_size=26)
            bottom = fitted_text(
                f"total reward {total_reward}    total regret {total_regret:.2f}",
                font_size=24,
            )
            bottom.set_color(GREY_A)
        else:
            top = fitted_text(f"t = {t}    {mode}    reward {reward}", font_size=26)
            added = 0.0 if regret_added < 1e-9 else regret_added
            bottom = fitted_text(
                f"total reward {total_reward}    regret +{added:.2f}    total regret {total_regret:.2f}",
                font_size=24,
            )
            bottom.set_color(ORANGE if added > 0 else GREY_A)
        top.set_color(mode_color)
        return VGroup(top, bottom).arrange(DOWN, buff=0.06)

    def swap_caption(self, old, words, font_size=30):
        new = fitted_text(words, font_size=font_size)
        new.to_edge(DOWN, buff=0.28)
        if old is None:
            self.play(FadeIn(new, shift=0.12 * UP), run_time=0.45)
        else:
            self.play(FadeTransform(old, new), run_time=0.45)
        return new

    def clear_screen(self):
        mobs = [mob for mob in self.mobjects if mob is not self.camera.frame]
        if not mobs:
            return
        self.play(*[FadeOut(mob) for mob in mobs], run_time=0.5)

    def define_the_problem(self):
        title = Text("The multi-armed bandit", font_size=46)
        title.to_edge(UP, buff=0.32)
        self.play(Write(title), run_time=0.9)

        machines = VGroup(*[
            make_machine(index, color)
            for index, color in enumerate(ARM_COLORS, start=1)
        ])
        machines.arrange(RIGHT, buff=0.7)
        machines.next_to(title, DOWN, buff=0.32)
        self.play(
            LaggedStart(*[FadeIn(machine, shift=0.15 * UP) for machine in machines], lag_ratio=0.18),
            run_time=1.3,
        )

        caption = self.swap_caption(
            None,
            "Three arms. Each has an unknown mean reward.",
        )
        self.wait(0.3)

        mu = Tex(R"\mu_i = \mathbb{E}\!\left[R_t \mid A_t = i\right]")
        mu.scale(0.95)
        mu.set_color(YELLOW)
        mu.next_to(machines, DOWN, buff=0.38)
        self.play(Write(mu), run_time=1.1)
        caption = self.swap_caption(
            caption,
            "Arm i pays reward R_t from a fixed distribution.",
        )
        self.wait(0.4)

        # One concrete round: the definition, filled in.
        chosen = machines[1]
        self.play(
            chosen.body.animate.set_stroke(WHITE, width=5),
            Indicate(chosen.button, color=WHITE, scale_factor=1.35),
            run_time=0.7,
        )
        self.play(
            chosen.readout.animate.set_opacity(0),
            chosen.reward.animate.set_opacity(1),
            run_time=0.4,
        )

        observed = Tex(R"t = 1,\quad A_1 = 2,\quad R_1 = 1")
        observed.scale(0.9)
        observed.next_to(mu, DOWN, buff=0.22)
        self.play(Write(observed), run_time=0.9)
        caption = self.swap_caption(
            caption,
            "The agent observes only the arm it selected.",
        )
        self.wait(0.6)

        regret = Tex(
            R"\mathcal{R}_T = T\mu^{*} - \mathbb{E}\!\left[\sum_{t=1}^{T} R_t\right]"
        )
        regret.scale(0.82)
        regret.move_to(VGroup(mu, observed))
        self.play(FadeOut(mu), FadeOut(observed), FadeIn(regret), run_time=0.8)
        caption = self.swap_caption(
            caption,
            "Regret is the reward lost against the best arm.",
        )
        self.wait(0.5)

        explore = VGroup(
            Text("Explore", font_size=34),
            fitted_text("try an uncertain arm", font_size=24),
        ).arrange(DOWN, buff=0.12)
        explore[0].set_color(ORANGE)
        explore[1].set_color(GREY_A)

        exploit = VGroup(
            Text("Exploit", font_size=34),
            fitted_text("pick the current best", font_size=24),
        ).arrange(DOWN, buff=0.12)
        exploit[0].set_color(GREEN)
        exploit[1].set_color(GREY_A)

        pair = VGroup(explore, exploit).arrange(RIGHT, buff=1.6)
        pair.move_to(regret)
        self.play(FadeOut(regret), FadeIn(pair), run_time=0.7)
        caption = self.swap_caption(
            caption,
            "Use it when the same choice repeats, and only the chosen action returns feedback.",
            font_size=26,
        )
        self.wait(1.1)

    def show_headline_example(self, pulls=None):
        if pulls is None:
            pulls = getattr(self, "pulls", PULLS)
        title = Text("Which headline gets the click?", font_size=40)
        title.to_edge(UP, buff=0.28)
        self.play(Write(title), run_time=0.8)

        rule = Tex(
            R"A_t = \arg\max_i \hat\mu_i \quad\text{with probability}\quad 1-\epsilon"
        )
        rule.scale(0.72)
        if rule.get_width() > 12.2:
            rule.set_width(12.2)
        rule.next_to(title, DOWN, buff=0.16)
        rule_note = fitted_text(
            "otherwise pick a headline at random",
            font_size=24,
        )
        rule_note.set_color(GREY_A)
        rule_note.next_to(rule, DOWN, buff=0.1)
        self.play(FadeIn(rule, shift=0.1 * UP), FadeIn(rule_note), run_time=0.7)

        cards = VGroup(*[
            make_card(name, color)
            for name, color in zip(HEADLINES, ARM_COLORS)
        ])
        cards.arrange(RIGHT, buff=0.42)
        cards.next_to(rule_note, DOWN, buff=0.28)
        self.play(
            LaggedStart(*[FadeIn(card, shift=0.12 * UP) for card in cards], lag_ratio=0.15),
            run_time=1.0,
        )

        mean_labels = []
        pull_labels = []
        true_labels = []
        for card, mean in zip(cards, TRUE_MEANS):
            mean_label = Text("mean  -", font_size=24)
            mean_label.set_color(card.arm_color)
            mean_label.next_to(card, DOWN, buff=0.16)
            pull_label = Text("pulls  0", font_size=22)
            pull_label.set_color(GREY_A)
            pull_label.next_to(mean_label, DOWN, buff=0.06)
            true_label = Text(f"true  {mean:.2f}", font_size=20)
            true_label.set_color(GREY_B)
            true_label.next_to(pull_label, DOWN, buff=0.06)
            mean_labels.append(mean_label)
            pull_labels.append(pull_label)
            true_labels.append(true_label)

        self.play(
            *[FadeIn(mob) for mob in [*mean_labels, *pull_labels, *true_labels]],
            run_time=0.4,
        )

        baseline_y = -2.15
        max_bar = 1.15
        axis = Line(
            np.array([-6.2, baseline_y, 0.0]),
            np.array([6.2, baseline_y, 0.0]),
        )
        axis.set_stroke(GREY_B, width=2)

        bars = []
        ticks = []
        for card, mean in zip(cards, TRUE_MEANS):
            x = card.get_center()[0]
            bar = Rectangle(width=0.46, height=0.05)
            bar.set_fill(card.arm_color, opacity=0.9)
            bar.set_stroke(width=0)
            bar.move_to(np.array([x, baseline_y, 0.0]), DOWN)
            bars.append(bar)

            tick = Line(0.42 * LEFT, 0.42 * RIGHT)
            tick.set_stroke(WHITE, width=3)
            tick.move_to(np.array([x, baseline_y + mean * max_bar, 0.0]))
            ticks.append(tick)

        zero = Text("0", font_size=18)
        zero.set_color(GREY_B)
        zero.next_to(axis.get_start(), LEFT, buff=0.12)
        one = Text("1", font_size=18)
        one.set_color(GREY_B)
        one.next_to(np.array([-6.2, baseline_y + max_bar, 0.0]), LEFT, buff=0.12)
        scale = Line(
            np.array([-6.2, baseline_y, 0.0]),
            np.array([-6.2, baseline_y + max_bar, 0.0]),
        )
        scale.set_stroke(GREY_B, width=2)

        legend = fitted_text("white mark = true click rate, hidden from the agent", font_size=22)
        legend.set_color(GREY_A)
        legend.next_to(axis, DOWN, buff=0.16)

        chart = VGroup(axis, scale, zero, one, *bars, *ticks, legend)
        self.play(FadeIn(chart), run_time=0.5)

        clicks = 0
        regret = 0.0
        successes = [0, 0, 0]
        counts = [0, 0, 0]
        status = self.scoreboard(0, "ready", None, 0, 0.0, 0.0)
        status.next_to(legend, DOWN, buff=0.12)
        self.play(FadeIn(status), run_time=0.3)

        previous = None
        leader = None
        for t, (arm, reward, mode) in enumerate(pulls, start=1):
            # The first dozen rounds are where the estimates swing the most.
            if t <= 12:
                clear_rt, pick_rt, reveal_rt, update_rt = 0.14, 0.22, 0.20, 0.42
            else:
                clear_rt, pick_rt, reveal_rt, update_rt = 0.08, 0.12, 0.10, 0.24
            card = cards[arm]
            if previous is not None:
                self.play(
                    previous.body.animate.set_stroke(previous.arm_color, width=2.5),
                    previous.blank.animate.set_opacity(1),
                    previous.click.animate.set_opacity(0),
                    previous.miss.animate.set_opacity(0),
                    run_time=clear_rt,
                )

            new_status = self.scoreboard(t, mode, None, clicks, 0.0, regret)
            new_status.move_to(status)
            self.play(
                card.body.animate.set_stroke(WHITE, width=5),
                FadeOut(status),
                FadeIn(new_status),
                run_time=pick_rt,
            )
            status = new_status

            self.play(
                card.blank.animate.set_opacity(0),
                card.click.animate.set_opacity(1 if reward else 0),
                card.miss.animate.set_opacity(0 if reward else 1),
                run_time=reveal_rt,
            )

            successes[arm] += reward
            counts[arm] += 1
            clicks += reward
            estimate = successes[arm] / counts[arm]
            delta = TRUE_MEANS[2] - TRUE_MEANS[arm]
            regret += delta

            new_mean = Text(f"mean  {estimate:.2f}", font_size=24)
            new_mean.set_color(card.arm_color)
            new_mean.move_to(mean_labels[arm])
            new_pull = Text(f"pulls  {counts[arm]}", font_size=22)
            new_pull.set_color(GREY_A)
            new_pull.move_to(pull_labels[arm])
            newer_status = self.scoreboard(t, mode, reward, clicks, delta, regret)
            newer_status.move_to(status)

            bar_height = max(0.05, estimate * max_bar)
            self.play(
                bars[arm].animate.set_height(bar_height, stretch=True, about_edge=DOWN),
                FadeTransform(mean_labels[arm], new_mean),
                FadeTransform(pull_labels[arm], new_pull),
                FadeOut(status),
                FadeIn(newer_status),
                run_time=update_rt,
            )
            mean_labels[arm] = new_mean
            pull_labels[arm] = new_pull
            status = newer_status
            previous = card
            estimates_now = [
                successes[i] / counts[i] if counts[i] else -1
                for i in range(3)
            ]
            new_leader = int(np.argmax(estimates_now))
            if leader is not None and new_leader != leader and t <= 16:
                self.wait(0.35)
            leader = new_leader

        self.wait(0.3)
        total = expected_regret(pulls)
        best_pulls = sum(1 for arm, _reward, _mode in pulls if arm == 2)
        formula = Tex(
            Rf"\mathcal{{R}} = \sum_t(\mu^{{*}} - \mu_{{A_t}}) = {total:.2f}"
        )
        formula.scale(0.7)
        closing = VGroup(
            fitted_text(
                f"Total reward {clicks}.   Regret {total:.2f}.   Best arm shown {best_pulls} of {len(pulls)}.",
                font_size=26,
            ),
            formula,
        )
        closing.arrange(DOWN, buff=0.12)
        # Keep the summary under the axis. A raw TeX box can be taller than
        # its glyphs, so cap the block before it climbs into the bars.
        closing.next_to(axis, DOWN, buff=0.18)
        if closing.get_height() > 0.95:
            closing.set_height(0.95)
            closing.next_to(axis, DOWN, buff=0.18)
        self.play(
            FadeOut(status),
            FadeOut(legend),
            FadeIn(closing),
            run_time=0.7,
        )
        self.wait(1.2)


class BanditDefinition(BanditScenes):
    def construct(self):
        self.define_the_problem()
        self.finish_batch_render()


class BanditUsageExample(BanditScenes):
    def construct(self):
        pulls = getattr(self, "pulls", PULLS)
        self.show_headline_example(pulls=pulls)
        self.finish_batch_render()


class UCBUsageExample(BanditUsageExample):
    """36-round headline optimization driven dynamically by UCB1."""
    pulls = generate_pull_trace("ucb", seed=16, rounds=36)


class ThompsonUsageExample(BanditUsageExample):
    """36-round headline optimization driven dynamically by Thompson Sampling."""
    pulls = generate_pull_trace("thompson", seed=16, rounds=36)


class BanditExplainer(BanditScenes):
    def construct(self):
        self.define_the_problem()
        self.clear_screen()
        pulls = getattr(self, "pulls", PULLS)
        self.show_headline_example(pulls=pulls)
        self.finish_batch_render()
