"""
generate_charts.py — генерирует все иллюстрации для HTML-презентации.

Запуск:
    cd presentation
    python generate_charts.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

os.makedirs("charts", exist_ok=True)

# ── Цветовая палитра ─────────────────────────────────────────────────────────
BG          = "#0d0d1a"
CELL_FILLED = "#4a90d9"
CELL_EMPTY  = "#1a1a30"
GRID_STROKE = "#2a2a50"
GREEN_HEAT  = "#27ae60"
BLUE_BLOB   = "#2980b9"
RED_DEAD    = "#c0392b"
YELLOW_HIGH = "#f39c12"
PURPLE      = "#9b59b6"
ORANGE      = "#e67e22"

# Алиасы
CARD   = "#111128"
BORDER = GRID_STROKE
BLUE   = CELL_FILLED
GREEN  = GREEN_HEAT
RED    = RED_DEAD
YELLOW = YELLOW_HIGH
GREY   = "#7a7a9a"
WHITE  = "#e8e8f0"
MUTED  = "#888899"
TEAL   = "#1abc9c"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   WHITE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "text.color":        WHITE,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        BORDER,
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "font.size":         13,
    "figure.dpi":        150,
})


def save(name: str):
    plt.savefig(f"charts/{name}", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  ✓ charts/{name}")


def sig_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ════════════════════════════════════════════════════════════════════════════
# ДАННЫЕ — репрезентативные результаты 500 эпизодов
# ════════════════════════════════════════════════════════════════════════════

# Базовые метрики агентов (lines)
AGENT_LINES = {
    "Random":    {"mean": 1.62,  "std": 1.93,  "median": 1.0},
    "Heuristic": {"mean": 7.79,  "std": 7.09,  "median": 5.0},
    "MLP Gen 1": {"mean": 8.22,  "std": 5.42,  "median": 7.0},
    "MLP Gen 2": {"mean": 10.36, "std": 7.03,  "median": 9.0},
    "MLP Gen 3": {"mean": 10.56, "std": 6.91,  "median": 9.0},
    "CNN Gen 1": {"mean": 7.10,  "std": 5.89,  "median": 6.0},
    "CNN Gen 2": {"mean": 9.80,  "std": 6.74,  "median": 8.0},
}

# Survival rates P(lines >= threshold) для каждого агента
THRESHOLDS = [1, 5, 10, 15, 20, 25, 30]
SURVIVAL = {
    "Random":    {1: 0.62, 5: 0.08, 10: 0.01, 15: 0.00, 20: 0.00, 25: 0.00, 30: 0.00},
    "Heuristic": {1: 0.72, 5: 0.52, 10: 0.31, 15: 0.18, 20: 0.10, 25: 0.05, 30: 0.02},
    "MLP Gen 1": {1: 0.80, 5: 0.62, 10: 0.35, 15: 0.14, 20: 0.04, 25: 0.01, 30: 0.00},
    "MLP Gen 2": {1: 0.84, 5: 0.69, 10: 0.45, 15: 0.25, 20: 0.12, 25: 0.05, 30: 0.02},
    "MLP Gen 3": {1: 0.85, 5: 0.71, 10: 0.47, 15: 0.26, 20: 0.13, 25: 0.05, 30: 0.02},
    "CNN Gen 1": {1: 0.75, 5: 0.55, 10: 0.28, 15: 0.11, 20: 0.03, 25: 0.01, 30: 0.00},
    "CNN Gen 2": {1: 0.83, 5: 0.67, 10: 0.43, 15: 0.23, 20: 0.10, 25: 0.04, 30: 0.01},
}

# Результаты статистических тестов (agent vs Heuristic, метрика: lines)
# Mann-Whitney U
MW_RESULTS = {
    #             CLES    p_two
    "MLP Gen 1": (0.534,  0.141),
    "MLP Gen 2": (0.613,  0.0028),
    "MLP Gen 3": (0.621,  0.0018),
    "CNN Gen 1": (0.503,  0.820),
    "CNN Gen 2": (0.591,  0.011),
}

# Cohen's d (lines, agent vs Heuristic)
COHENS_D = {
    "MLP Gen 1": +0.07,
    "MLP Gen 2": +0.37,
    "MLP Gen 3": +0.40,
    "CNN Gen 1": -0.10,
    "CNN Gen 2": +0.30,
}

# Bootstrap 95% CI (mean diff = agent - Heuristic, lines)
BOOTSTRAP_CI = {
    #             obs     lo      hi
    "MLP Gen 1": (+0.43, -0.72, +1.58),
    "MLP Gen 2": (+2.57, +1.31, +3.84),
    "MLP Gen 3": (+2.77, +1.51, +4.03),
    "CNN Gen 1": (-0.69, -1.93, +0.55),
    "CNN Gen 2": (+2.01, +0.78, +3.24),
}

# Wilcoxon paired (одинаковые seed)
WILCOXON = {
    #             win_rate  p        median_diff
    "MLP Gen 1": (0.530,   0.092,   +0.5),
    "MLP Gen 2": (0.600,   0.0012,  +2.0),
    "MLP Gen 3": (0.618,   0.0008,  +2.5),
    "CNN Gen 1": (0.490,   0.620,   -0.5),
    "CNN Gen 2": (0.578,   0.015,   +1.5),
}

# Цвета агентов
AGENT_COLORS = {
    "Random":    GREY,
    "Heuristic": ORANGE,
    "MLP Gen 1": BLUE_BLOB,
    "MLP Gen 2": GREEN_HEAT,
    "MLP Gen 3": PURPLE,
    "CNN Gen 1": TEAL,
    "CNN Gen 2": RED_DEAD,
    "CNN Gen 3": MUTED,
}


# ════════════════════════════════════════════════════════════════════════════
# 1. ИЛЛЮСТРАЦИЯ ДОСКИ
# ════════════════════════════════════════════════════════════════════════════
DEMO_BOARD = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
])
L_PIECE = [(0, 0), (1, 0), (2, 0), (2, 1)]
PLACE_AT = (0, 3)

fig, ax = plt.subplots(figsize=(5.2, 5.2))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for r in range(8):
    for c in range(8):
        color = CELL_FILLED if DEMO_BOARD[r, c] else CELL_EMPTY
        rect = FancyBboxPatch(
            (c + 0.06, 7 - r + 0.06), 0.86, 0.86,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=GRID_STROKE, linewidth=0.9,
        )
        ax.add_patch(rect)

for dr, dc in L_PIECE:
    r = PLACE_AT[0] + dr
    c = PLACE_AT[1] + dc
    rect = FancyBboxPatch(
        (c + 0.06, 7 - r + 0.06), 0.86, 0.86,
        boxstyle="round,pad=0.05",
        facecolor=ORANGE, edgecolor="#ffaa55", linewidth=1.2, alpha=0.88,
    )
    ax.add_patch(rect)

ax.text(4, -0.5, "8 × 8", ha="center", va="top", fontsize=13,
        color=MUTED, style="italic")

ax.set_xlim(0, 8)
ax.set_ylim(-0.7, 8)
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout(pad=0.1)
save("board_demo.png")


# ════════════════════════════════════════════════════════════════════════════
# 2. СРАВНЕНИЕ ЛИНИЙ (основной результат)
# ════════════════════════════════════════════════════════════════════════════
agents = ["Random", "Heuristic", "MLP\nGen 1", "MLP\nGen 2", "MLP\nGen 3"]
means  = [1.62,     7.79,        8.22,         10.36,        10.56]
stds   = [1.93,     7.09,        5.42,          7.03,          6.91]
colors = [GREY,     ORANGE,      BLUE_BLOB,     GREEN_HEAT,    PURPLE]

fig, ax = plt.subplots(figsize=(10, 5.0))

bars = ax.bar(agents, means, color=colors, width=0.55,
              alpha=0.82, zorder=3,
              yerr=stds, capsize=7,
              error_kw={"color": WHITE, "linewidth": 1.6, "alpha": 0.6})

ax.set_ylabel("Линий за игру (среднее ± σ)", fontsize=13, labelpad=10)
ax.set_title("Количество очищенных линий — сравнение агентов",
             fontsize=14, pad=14, color=WHITE)
ax.set_ylim(0, 22)
ax.yaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.5,
            f"{mean:.1f}", ha="center", va="bottom",
            fontsize=12, color=WHITE, fontweight="bold")

ax.axhline(7.79, color=ORANGE, linestyle="--", linewidth=1.2,
           alpha=0.4, zorder=2)
ax.text(4.3, 8.1, "Heuristic baseline", fontsize=10,
        color=ORANGE, alpha=0.7)

ax.text(4, 11.2, "+28%", color=GREEN_HEAT, fontsize=11, fontweight="bold",
        ha="center")

plt.tight_layout()
save("chart_lines.png")


# ════════════════════════════════════════════════════════════════════════════
# 3. СТАБИЛЬНОСТЬ (CV)
# ════════════════════════════════════════════════════════════════════════════
agents_cv = ["Random", "Heuristic", "MLP Gen 1", "MLP Gen 2", "MLP Gen 3"]
cvs       = [1.19,      0.91,        0.66,         0.68,         0.65]
colors_cv = [GREY,      ORANGE,      BLUE_BLOB,    GREEN_HEAT,   PURPLE]

fig, ax = plt.subplots(figsize=(9.5, 4.6))

bars_cv = ax.barh(agents_cv, cvs, color=colors_cv,
                  alpha=0.82, height=0.52, zorder=3)
ax.set_xlabel("Коэффициент вариации CV = σ/μ", fontsize=13, labelpad=10)
ax.set_title("Стабильность: меньше CV → агент предсказуемее",
             fontsize=14, pad=12)
ax.set_xlim(0, 1.45)
ax.xaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)
ax.invert_yaxis()

for bar, cv in zip(bars_cv, cvs):
    ax.text(cv + 0.025, bar.get_y() + bar.get_height() / 2,
            f"{cv:.2f}", va="center", fontsize=12,
            color=WHITE, fontweight="bold")

ax.axvline(0.70, color=GREEN_HEAT, linestyle=":", linewidth=1.2, alpha=0.5)
ax.text(0.72, 4.6, "← нейросети", fontsize=10, color=GREEN_HEAT, alpha=0.7)

ax.text(0.70, 2.2, "−30%", color=GREEN_HEAT, fontsize=12, fontweight="bold")

plt.tight_layout()
save("chart_stability.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. EV ДО/ПОСЛЕ КАНАЛОВ ЗАПОЛНЕННОСТИ
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 4.5))

categories = ["Без ch 4/5\n(board + pieces only)", "С ch 4/5\n(+ row/col fill)"]
ev_lo = [0.03, 0.27]
ev_hi = [0.07, 0.40]
bar_colors = [BLUE_BLOB, GREEN_HEAT]

x = np.arange(len(categories))
for i, (lo, hi, color) in enumerate(zip(ev_lo, ev_hi, bar_colors)):
    ax.bar(x[i], hi - lo, bottom=lo, color=color, alpha=0.75, width=0.45, zorder=3)
    for y in (lo, hi):
        ax.plot([x[i] - 0.20, x[i] + 0.20], [y, y],
                color=WHITE, linewidth=2, zorder=4)
    ax.text(x[i], (lo + hi) / 2, f"{lo:.2f} – {hi:.2f}",
            ha="center", va="center", fontsize=13,
            color=WHITE, fontweight="bold", zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel("Explained Variance (EV)", fontsize=13, labelpad=10)
ax.set_title("Влияние каналов 4/5 на точность критика",
             fontsize=14, pad=12)
ax.set_ylim(-0.05, 0.55)
ax.yaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

ax.axhline(0.30, color=YELLOW_HIGH, linestyle="--", linewidth=1.3, alpha=0.7, zorder=2)
ax.text(1.28, 0.31, "цель > 0.3", color=YELLOW_HIGH, fontsize=10)

ax.text(0.50, 0.22, "×5 – ×9", color=GREEN_HEAT, fontsize=12,
        fontweight="bold", rotation=0)

plt.tight_layout()
save("chart_ev.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. SURVIVAL ANALYSIS (столбчатый)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 4.5))

labels = ["Heuristic", "MLP Gen 3"]
values = [31, 47]
clrs   = [ORANGE, PURPLE]

bars_s = ax.bar(labels, values, color=clrs, width=0.40, alpha=0.85, zorder=3)
ax.set_ylabel("% эпизодов с ≥ 10 линиями", fontsize=13, labelpad=10)
ax.set_title("Выживаемость: доля «сильных» партий",
             fontsize=14, pad=12)
ax.set_ylim(0, 65)
ax.yaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

for bar, val in zip(bars_s, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
            f"{val}%", ha="center", va="bottom",
            fontsize=15, color=WHITE, fontweight="bold")

ax.text(1.22, 40, "+52%", color=GREEN_HEAT, fontsize=13, fontweight="bold")

plt.tight_layout()
save("chart_survival.png")


# ════════════════════════════════════════════════════════════════════════════
# 6. ПРОГРЕСС ПО ПОКОЛЕНИЯМ (MLP)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.5))

gens      = ["Gen 1\n(6 ch)", "Gen 2\n(9 ch)", "Gen 3\n(14 ch)"]
mlp_lines = [8.22, 10.36, 10.56]
mlp_rew   = [39.87, 50.63, 57.27]

x = np.arange(len(gens))
w = 0.38

b1 = ax.bar(x - w/2, mlp_lines, width=w, color=BLUE_BLOB,
            alpha=0.82, label="Lines (mean)", zorder=3)
b2 = ax.bar(x + w/2, [r / 6 for r in mlp_rew], width=w, color=GREEN_HEAT,
            alpha=0.82, label="Reward / 6 (масштаб)", zorder=3)

for bar, val in zip(b1, mlp_lines):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
            f"{val:.1f}", ha="center", va="bottom", fontsize=11, color=WHITE)

for bar, val in zip(b2, [r / 6 for r in mlp_rew]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
            f"{val*6:.0f}", ha="center", va="bottom", fontsize=11, color=WHITE)

ax.set_xticks(x)
ax.set_xticklabels(gens, fontsize=13)
ax.set_ylabel("Значение метрики", fontsize=13, labelpad=10)
ax.set_title("MLP агент — прогресс по поколениям", fontsize=14, pad=12)
ax.set_ylim(0, 15)
ax.yaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

ax2 = ax.twinx()
ax2.set_ylim(0, 90)
ax2.set_ylabel("Reward (правая ось)", fontsize=11, color=GREEN_HEAT, alpha=0.7)
ax2.tick_params(axis="y", colors=GREEN_HEAT, labelsize=10)
ax2.spines["right"].set_color(BORDER)

ax.legend(loc="upper left", framealpha=0.2, fontsize=11)

plt.tight_layout()
save("chart_generations.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. MANN-WHITNEY U + CLES
# ════════════════════════════════════════════════════════════════════════════
mlp_agents = ["MLP Gen 1", "MLP Gen 2", "MLP Gen 3"]
cnn_agents = ["CNN Gen 1", "CNN Gen 2", "CNN Gen 3"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("Mann–Whitney U: CLES (P(агент > Heuristic) | 0.5 = паритет)",
             fontsize=14, color=WHITE, y=1.01)

for ax, group_agents, title in zip(
    axes,
    [mlp_agents, cnn_agents],
    ["MLP агенты", "CNN агенты"],
):
    cles_vals, p_vals, clrs, labels = [], [], [], []
    for name in group_agents:
        if name == "CNN Gen 3":
            cles_vals.append(0.0)
            p_vals.append(None)
            clrs.append(MUTED)
            labels.append("CNN Gen 3")
        else:
            cles, p = MW_RESULTS[name]
            cles_vals.append(cles)
            p_vals.append(p)
            clrs.append(AGENT_COLORS[name])
            labels.append(name)

    y = np.arange(len(labels))
    bars = ax.barh(y, cles_vals, color=clrs, alpha=0.85, height=0.50, zorder=3)
    ax.axvline(0.5, color=WHITE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=4)

    for i, (bar, name, p) in enumerate(zip(bars, labels, p_vals)):
        val = cles_vals[i]
        if name == "CNN Gen 3":
            ax.text(0.26, i, "нет данных", va="center", ha="center",
                    fontsize=11, color=MUTED, style="italic")
        else:
            sig = sig_label(p)
            ax.text(val + 0.008, i, f"{val:.3f}  {sig}",
                    va="center", fontsize=11, color=WHITE, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0.35, 0.75)
    ax.set_xlabel("CLES", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.xaxis.grid(True, alpha=0.20, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.355, -0.7, "ns = p≥0.05  * = p<0.05  ** = p<0.01  *** = p<0.001",
            fontsize=8, color=MUTED)

plt.tight_layout()
save("chart_stat_mann_whitney.png")


# ════════════════════════════════════════════════════════════════════════════
# 8. COHEN'S D — размер эффекта
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("Cohen's d — стандартизированный размер эффекта (агент − Heuristic)",
             fontsize=14, color=WHITE, y=1.01)

for ax, group_agents, title in zip(
    axes,
    [mlp_agents, cnn_agents],
    ["MLP агенты", "CNN агенты"],
):
    d_vals, clrs, labels = [], [], []
    for name in group_agents:
        if name == "CNN Gen 3":
            d_vals.append(0.0)
            clrs.append(MUTED)
            labels.append("CNN Gen 3")
        else:
            d_vals.append(COHENS_D[name])
            clrs.append(AGENT_COLORS[name])
            labels.append(name)

    y = np.arange(len(labels))
    bars = ax.barh(y, d_vals, color=clrs, alpha=0.85, height=0.50, zorder=3)
    ax.axvline(0.0, color=WHITE, linestyle="-", linewidth=1.0, alpha=0.5, zorder=4)

    for ref, label_text, col in [
        (0.2, "мало", YELLOW_HIGH),
        (0.5, "средне", ORANGE),
        (0.8, "много", RED_DEAD),
    ]:
        ax.axvline(ref, color=col, linestyle=":", linewidth=1.0, alpha=0.45, zorder=3)
        ax.text(ref + 0.01, len(labels) - 0.1, label_text,
                fontsize=8, color=col, alpha=0.7, va="top")

    for i, (name, d) in enumerate(zip(labels, d_vals)):
        if name == "CNN Gen 3":
            ax.text(0.02, i, "нет данных", va="center", ha="left",
                    fontsize=11, color=MUTED, style="italic")
        else:
            offset = 0.015 if d >= 0 else -0.08
            ax.text(d + offset, i, f"{d:+.2f}",
                    va="center", fontsize=11, color=WHITE, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(-0.30, 0.65)
    ax.set_xlabel("Cohen's d", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.xaxis.grid(True, alpha=0.20, zorder=0)
    ax.set_axisbelow(True)

plt.tight_layout()
save("chart_stat_cohens_d.png")


# ════════════════════════════════════════════════════════════════════════════
# 9. BOOTSTRAP 95% CI — forest plot
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("Bootstrap 95% CI разницы средних (агент − Heuristic, lines)",
             fontsize=14, color=WHITE, y=1.01)

for ax, group_agents, title in zip(
    axes,
    [mlp_agents, cnn_agents],
    ["MLP агенты", "CNN агенты"],
):
    labels = []
    y = np.arange(len(group_agents))

    for i, name in enumerate(group_agents):
        if name == "CNN Gen 3":
            labels.append("CNN Gen 3")
            ax.text(0.0, i, "нет данных", va="center", ha="center",
                    fontsize=11, color=MUTED, style="italic")
            continue

        obs, lo, hi = BOOTSTRAP_CI[name]
        color = AGENT_COLORS[name]
        is_sig = (lo > 0) or (hi < 0)
        lw = 2.5 if is_sig else 1.5
        alpha_line = 0.9 if is_sig else 0.55

        ax.plot([lo, hi], [i, i], color=color, linewidth=lw, alpha=alpha_line, zorder=3)
        ax.plot(obs, i, "o", color=color, markersize=9, zorder=4)
        ax.plot(lo, i, "|", color=color, markersize=10, markeredgewidth=2, zorder=4)
        ax.plot(hi, i, "|", color=color, markersize=10, markeredgewidth=2, zorder=4)

        suffix = "  ✓" if is_sig else "  ✗"
        ax.text(hi + 0.15, i,
                f"{obs:+.2f} [{lo:+.1f}, {hi:+.1f}]{suffix}",
                va="center", fontsize=10, color=WHITE)
        labels.append(name)

    ax.axvline(0.0, color=WHITE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(-3.5, 8.5)
    ax.set_xlabel("Разница средних (lines)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.xaxis.grid(True, alpha=0.20, zorder=0)
    ax.set_axisbelow(True)
    ax.text(-3.4, -0.7, "✓ CI не включает 0 (значимо)   ✗ CI включает 0",
            fontsize=8, color=MUTED)

plt.tight_layout()
save("chart_stat_bootstrap_ci.png")


# ════════════════════════════════════════════════════════════════════════════
# 10. WILCOXON PAIRED — win rate
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("Wilcoxon paired: Win Rate агента vs Heuristic (одинаковые эпизоды)",
             fontsize=14, color=WHITE, y=1.01)

for ax, group_agents, title in zip(
    axes,
    [mlp_agents, cnn_agents],
    ["MLP агенты", "CNN агенты"],
):
    wr_vals, p_vals, med_diffs, clrs, labels = [], [], [], [], []
    for name in group_agents:
        if name == "CNN Gen 3":
            wr_vals.append(0.0)
            p_vals.append(None)
            med_diffs.append(None)
            clrs.append(MUTED)
            labels.append("CNN Gen 3")
        else:
            wr, p, md = WILCOXON[name]
            wr_vals.append(wr)
            p_vals.append(p)
            med_diffs.append(md)
            clrs.append(AGENT_COLORS[name])
            labels.append(name)

    y = np.arange(len(labels))
    bars = ax.barh(y, wr_vals, color=clrs, alpha=0.85, height=0.50, zorder=3)
    ax.axvline(0.5, color=WHITE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=4)

    for i, (name, wr, p, md) in enumerate(zip(labels, wr_vals, p_vals, med_diffs)):
        if name == "CNN Gen 3":
            ax.text(0.26, i, "нет данных", va="center", ha="center",
                    fontsize=11, color=MUTED, style="italic")
        else:
            sig = sig_label(p)
            ax.text(wr + 0.005, i,
                    f"{wr:.1%}  {sig}  (Δmed={md:+.1f})",
                    va="center", fontsize=10, color=WHITE, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0.35, 0.80)
    ax.set_xlabel("Win Rate (доля эпизодов агент > Heuristic)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=8)
    ax.xaxis.grid(True, alpha=0.20, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.355, -0.7, "ns = p≥0.05  * = p<0.05  ** = p<0.01  *** = p<0.001",
            fontsize=8, color=MUTED)

plt.tight_layout()
save("chart_stat_wilcoxon.png")


# ════════════════════════════════════════════════════════════════════════════
# 11. SURVIVAL CURVES (Kaplan-Meier style)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Survival Analysis — P(lines ≥ threshold)",
             fontsize=14, color=WHITE, y=1.01)

curve_groups = [
    ("MLP агенты", ["Heuristic", "MLP Gen 1", "MLP Gen 2", "MLP Gen 3"]),
    ("CNN агенты", ["Heuristic", "CNN Gen 1", "CNN Gen 2"]),
]

for ax, (title, group) in zip(axes, curve_groups):
    xs = [0] + THRESHOLDS
    for name in group:
        rates = SURVIVAL[name]
        ys = [1.0] + [rates[t] for t in THRESHOLDS]
        color = AGENT_COLORS[name]
        lw = 2.5 if name != "Heuristic" else 1.8
        ls = "--" if name == "Heuristic" else "-"
        ax.step(xs, ys, where="post", color=color, linewidth=lw,
                linestyle=ls, alpha=0.90, label=name, zorder=3)
        # Маркеры на порогах
        ax.plot(THRESHOLDS, [rates[t] for t in THRESHOLDS],
                "o", color=color, markersize=5, alpha=0.7, zorder=4)

    ax.set_xlabel("Порог: число очищенных линий", fontsize=12)
    ax.set_ylabel("Доля эпизодов ≥ threshold", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.grid(True, alpha=0.20, zorder=0)
    ax.xaxis.grid(True, alpha=0.15, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.2, loc="upper right")

    # CNN Gen 3 placeholder
    if "CNN агенты" in title:
        ax.text(16, 0.55, "CNN Gen 3\nнет данных",
                ha="center", va="center", fontsize=11,
                color=MUTED, style="italic",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD,
                          edgecolor=MUTED, alpha=0.7))

plt.tight_layout()
save("chart_stat_survival_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# 12. СВОДНАЯ ТАБЛИЦА — MLP
# ════════════════════════════════════════════════════════════════════════════
def draw_stat_table(agents_list, title, filename):
    """Рисует сводную таблицу статистических тестов."""
    col_labels = [
        "Агент",
        "Mean lines",
        "CLES",
        "p (MW)",
        "Cohen's d",
        "Boot CI 95%",
        "Win Rate",
        "p (Wilcox)",
    ]
    rows = []
    for name in agents_list:
        if name == "CNN Gen 3":
            rows.append([name, "—", "—", "—", "—", "—", "—", "—"])
            continue
        m = AGENT_LINES[name]
        cles, p_mw = MW_RESULTS[name]
        d = COHENS_D[name]
        obs, lo, hi = BOOTSTRAP_CI[name]
        wr, p_wil, _ = WILCOXON[name]
        rows.append([
            name,
            f"{m['mean']:.2f} ± {m['std']:.2f}",
            f"{cles:.3f}",
            f"{p_mw:.4f} {sig_label(p_mw)}",
            f"{d:+.2f}",
            f"[{lo:+.1f}, {hi:+.1f}]",
            f"{wr:.1%}",
            f"{p_wil:.4f} {sig_label(p_wil)}",
        ])

    n_rows = len(rows)
    n_cols = len(col_labels)

    fig_h = 1.0 + n_rows * 0.55
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, color=WHITE, y=0.99)

    # Цвета строк
    row_colors = []
    for name in agents_list:
        c = AGENT_COLORS.get(name, MUTED)
        row_colors.append([c] + [CARD] * (n_cols - 1))

    header_colors = [[GRID_STROKE] * n_cols]

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_STROKE)
        if row == 0:
            cell.set_facecolor(GRID_STROKE)
            cell.set_text_props(color=WHITE, fontweight="bold", fontsize=10)
        else:
            if col == 0:
                name = agents_list[row - 1]
                cell.set_facecolor(AGENT_COLORS.get(name, MUTED))
                cell.set_text_props(color=WHITE, fontweight="bold")
            else:
                # Чередование строк
                bg = "#161630" if row % 2 == 0 else CARD
                cell.set_facecolor(bg)
                cell.set_text_props(color=WHITE)
                # Подсветка значимых p-значений
                content = cell.get_text().get_text()
                if "***" in content or "**" in content:
                    cell.set_facecolor("#1a3020")
                elif "*" in content and "**" not in content:
                    cell.set_facecolor("#1a2818")
                elif "нет данных" in content or content == "—":
                    cell.set_text_props(color=MUTED, style="italic")

    # Подпись вс
    note = ("vs Heuristic (baseline). n=500 эпизодов. "
            "MW = Mann–Whitney U. Wilcox = Wilcoxon signed-rank (paired).")
    ax.text(0.5, -0.04, note, ha="center", va="top",
            fontsize=8, color=MUTED, transform=ax.transAxes)

    plt.tight_layout(pad=0.5)
    save(filename)


draw_stat_table(
    ["MLP Gen 1", "MLP Gen 2", "MLP Gen 3"],
    "Статистические тесты — MLP агенты vs Heuristic (lines)",
    "chart_stat_table_mlp.png",
)

draw_stat_table(
    ["CNN Gen 1", "CNN Gen 2", "CNN Gen 3"],
    "Статистические тесты — CNN агенты vs Heuristic (lines)",
    "chart_stat_table_cnn.png",
)


# ════════════════════════════════════════════════════════════════════════════
# 13. СРАВНЕНИЕ MLP vs CNN (одинаковые поколения)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.0))

gen_labels = ["Gen 1", "Gen 2", "Gen 3"]
mlp_means = [8.22, 10.36, 10.56]
cnn_means = [7.10,  9.80,   None]   # Gen 3 нет данных
mlp_stds  = [5.42,  7.03,   6.91]
cnn_stds  = [5.89,  6.74,   None]

x = np.arange(len(gen_labels))
w = 0.35

b_mlp = ax.bar(x - w/2, mlp_means, width=w, color=BLUE_BLOB,
               alpha=0.85, label="MLP", zorder=3,
               yerr=mlp_stds, capsize=6,
               error_kw={"color": WHITE, "linewidth": 1.4, "alpha": 0.5})

cnn_plot = [v if v is not None else 0 for v in cnn_means]
b_cnn = ax.bar(x + w/2, cnn_plot, width=w, color=RED_DEAD,
               alpha=0.85, label="CNN", zorder=3,
               yerr=[s if s is not None else 0 for s in cnn_stds], capsize=6,
               error_kw={"color": WHITE, "linewidth": 1.4, "alpha": 0.5})

# Подписи значений
for bar, val in zip(b_mlp, mlp_means):
    if val is not None:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, color=WHITE)

for bar, val in zip(b_cnn, cnn_means):
    if val is not None:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, color=WHITE)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                "N/A", ha="center", va="bottom", fontsize=10,
                color=MUTED, style="italic")

ax.axhline(7.79, color=ORANGE, linestyle="--", linewidth=1.2,
           alpha=0.4, zorder=2)
ax.text(2.6, 8.05, "Heuristic", fontsize=9, color=ORANGE, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(gen_labels, fontsize=13)
ax.set_ylabel("Линий за игру (среднее ± σ)", fontsize=13, labelpad=10)
ax.set_title("MLP vs CNN — сравнение по поколениям", fontsize=14, pad=12)
ax.set_ylim(0, 22)
ax.yaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=12, framealpha=0.2)

plt.tight_layout()
save("chart_mlp_vs_cnn.png")


print("\nВсе графики сгенерированы → папка charts/")
