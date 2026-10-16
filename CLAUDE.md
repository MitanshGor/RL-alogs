# CLAUDE.md — System Instructions & Guardrails for RL-BLOG-ANIMATION

> **Project Overview:**  
> `RL-BLOG-ANIMATION` is an educational, mathematically rigorous, and visually stunning animated series and blog explaining Reinforcement Learning (RL) concepts from first principles. Built with **3Blue1Brown's ManimGL (v1.7.2)**, Python 3.11, LaTeX, and NumPy.

---

## 1. Environment & Tech Stack

- **Language:** Python 3.11+
- **Core Engine:** ManimGL (`manimgl==1.7.2`) via PyOpenGL / ModernGL / pyglet
- **Virtual Environment:** `.venv/` at workspace root
- **Math & Computing:** NumPy 2.x, SymPy, SciPy
- **Typesetting & Fonts:**
  - Mac OS X environment: Font set to `"Helvetica"` in `custom_config.yml` (since Windows Consolas is unavailable).
  - LaTeX template: `"basic"` override in `custom_config.yml` to prevent 3b1b package crashes (avoids missing `tipa`, `calligra`, `physics`).
- **Media Output:** H.264 / MP4 via FFmpeg into `videos/` subdirectories.

---

## 2. Command Palette & Workflow

Always activate the local virtual environment before executing commands:
```bash
source .venv/bin/activate
```

### Headless Terminal Simulation (Zero Graphics)
Run pure Python simulations from workspace root or module directory:
```bash
# Compare all four algorithms (Epsilon-Greedy, UCB1, Thompson Sampling, Softmax)
python3 simulate.py

# Verbose step-by-step trace of 36 rounds
python3 simulate.py --algo epsilon-greedy -v

# Extended benchmark (1,000 rounds)
python3 simulate.py --rounds 1000

# Generate all 6 empirical research plots & master dashboard
python3 plot_comparisons.py
```

### ManimGL Rendering Commands
Run these from the specific topic folder (e.g. `cd Multiarmbandit`):
```bash
# 1. Interactive Preview (opens real-time OpenGL window; press 'q' or 'Esc' to exit)
manimgl multiarmed_bandit.py BanditDefinition

# 2. Fast Low-Res Preview (480p, useful for checking animations quickly)
manimgl multiarmed_bandit.py BanditDefinition -l

# 3. High-Speed Frame Snapshot (saves single image of final frame without full render)
manimgl multiarmed_bandit.py BanditDefinition -s

# 4. Full Production Render (writes .mp4 to ./videos/)
manimgl multiarmed_bandit.py BanditExplainer -w

# 5. Medium 720p Render
manimgl multiarmed_bandit.py BanditUsageExample -m -w
```

### Linting & Code Verification
```bash
python3 -m py_compile multiarmed_bandit.py
```

---

## 3. Coding Standards & Guardrails for AI

### ManimGL Best Practices
1. **Raw Strings for LaTeX:** Always prefix TeX expressions with `r` (e.g. `Tex(r"\mu_i = \mathbb{E}[R_t \mid A_t = i]")`) to prevent invalid escape sequence warnings.
2. **Tex vs Text:**
   - Use `Tex` or `MathTex` for formulas, math symbols, and equations.
   - Use `Text` for UI labels, titles, and explanations with clean sans-serif typography.
3. **Color Palette Consistency:**
   - Use ManimGL's curated constants: `BLUE_C`, `TEAL_C`, `GOLD_C`, `GREY_A`, `RED_C`, `GREEN_C`.
   - Never hardcode arbitrary hex values unless defining a documented theme token.
4. **Coordinate Positioning:**
   - Anchor elements relative to other elements using `.next_to()`, `.align_to()`, and directions (`UP`, `DOWN`, `LEFT`, `RIGHT`).
   - Avoid absolute screen coordinates unless placing fixed frame headers or HUD elements.
5. **Animation Pacing & Timing:**
   - Scene transitions should feel smooth and natural: default `run_time=1.0` to `1.5`, with purposeful `self.wait(1.0)` pauses for readability.
   - Group related animations using `LaggedStart` or `AnimationGroup` to prevent visual overload.

### Simulation & Math Guardrails
- **Deterministic Demonstrations:** Always fix random seeds (e.g. `seed=16` in multi-armed bandit trace) so animated traces, regret counts, and running means are 100% reproducible.
- **Instantaneous vs Cumulative Regret:** Strictly adhere to theoretical RL formulations ($\mu^* - \mu_{A_t}$ for instantaneous; $\mathcal{R}_T = \sum (\mu^* - \mu_{A_t})$ for cumulative).
- **Separation of Concerns:** Keep raw simulation algorithms (e.g., epsilon-greedy, UCB, Thompson Sampling) isolated from Manim visual mobjects so they can be unit-tested without launching OpenGL windows.

---

## 4. Multi-Module Project Structure

```text
RL-BLOG-ANIMATION/
├── CLAUDE.md              # System instructions & AI guidelines (this file)
├── .cursorrules           # Cursor IDE agent instructions
├── AGENTS.md              # Multi-agent roles & strict boundaries
├── PRD.md                 # Product requirements & curriculum roadmap
├── TECH_SPEC.md           # Architecture, rendering pipelines & math spec
├── CHANGELOG.md           # Version histories & iteration logs
├── CONTRIBUTING.md        # Vibe-coding collaboration rules & commit conventions
├── .venv/                 # Python 3.11 virtualenv
└── Multiarmbandit/        # Module 01: Multi-Armed Bandits
    ├── multiarmed_bandit.py # ManimGL animation scenes
    ├── custom_config.yml    # Mac/Helvetica & TeX configuration override
    ├── README.md            # Deep-dive math explainer & article
    ├── requirements.txt     # Module dependencies
    └── videos/              # Rendered MP4 files (BanditExplainer.mp4, etc.)
```

---

## 5. Vibe-Coding Rules of Engagement

1. **Never Render Full Videos Unprompted:** Video rendering is compute-intensive and can lock the GPU. Always suggest `-s` (snapshot) or `-l` (low-res preview) during iterative development before writing `-w`.
2. **Never Break LaTeX Formatting:** Test TeX formulas against the `basic` template. Do not introduce unsupported CTAN packages into equations.
3. **Keep Explanations Intuitive:** Every scene must answer: *What is the intuition? What is the mathematical form? How does the agent act in practice?*
