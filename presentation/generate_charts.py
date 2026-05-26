"""
generate_charts.py — генерирует все иллюстрации для HTML-презентации.

Запуск:
    python presentation/generate_charts.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import shutil

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(_PROJECT_ROOT, "results", "charts")
PICTURES_DIR = os.path.join(_PROJECT_ROOT, "presentation", "tex", "pictures")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)

# Убираем устаревшие файлы
for _old in ["chart_stability.png", "chart_survival.png", "chart_generations.png",
             "board_demo.png"]:
    _p = os.path.join(CHARTS_DIR, _old)
    if os.path.exists(_p):
        os.remove(_p)
        print(f"  ✗ удалён {_p}")

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

CARD   = "#111128"
BORDER = GRID_STROKE
BLUE   = CELL_FILLED
GREEN  = GREEN_HEAT
RED    = RED_DEAD
YELLOW = YELLOW_HIGH
TEAL   = "#1abc9c"
GREY   = "#7a7a9a"
WHITE  = "#e8e8f0"
MUTED  = "#888899"

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
    path = os.path.join(CHARTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    shutil.copy2(path, os.path.join(PICTURES_DIR, name))
    print(f"  ✓ results/charts/{name} → presentation/tex/pictures/{name}")


def sig_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def draw_cell(ax, col, row_y, color, border=GRID_STROKE, lw=0.9, alpha=1.0):
    """Рисует одну ячейку сетки."""
    rect = FancyBboxPatch(
        (col + 0.06, row_y + 0.06), 0.86, 0.86,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=border, linewidth=lw, alpha=alpha,
    )
    ax.add_patch(rect)


def draw_piece(ax, shape, origin_col, origin_row_y, color, border="#ffaa55",
               lw=1.2, alpha=0.90):
    """Рисует фигуру (numpy 2D array) с верхним-левым углом в origin."""
    for r in range(shape.shape[0]):
        for c in range(shape.shape[1]):
            if shape[r, c]:
                draw_cell(ax, origin_col + c, origin_row_y - r,
                          color, border=border, lw=lw, alpha=alpha)


# ── Лог-данные ───────────────────────────────────────────────────────────────
def _log(rel: str) -> str:
    return os.path.join(_PROJECT_ROOT, rel)

LOG_PATHS = {
    "MLP Gen 1": _log("models/gen1/mlp_gen_1/training_console.log"),
    "MLP Gen 2": _log("models/gen2/mlp_gen_2/training_console.log"),
    "MLP Gen 3": _log("models/gen3/mlp_gen_3/training_console.log"),
    "CNN Gen 1": _log("models/gen1/cnn_gen_1/training_console.log"),
    "CNN Gen 2": _log("models/gen2/cnn_gen_2/training_console.log"),
    "CNN Gen 3": _log("models/gen3/cnn_gen_3/training_console.log"),
    "CNN Gen 3 TR": _log("models/gen3/cnn_gen_3_tr/training_console.log"),
    "CNN Gen 4": _log("models/gen4/cnn/notebookc5318547d8.log"),
    "ViT Gen 4": _log("models/gen4/vit/notebook7a01605639.log"),
}

_STEPS_RE = re.compile(r'\|\s*total_timesteps\s*\|\s*([\d.e+]+)\s*\|')
_REW_RE   = re.compile(r'\|\s*ep_rew_mean\s*\|\s*([\d.e+\-]+)\s*\|')


def parse_log(path):
    log = open(path).read()
    steps = np.array([float(m.group(1)) for m in _STEPS_RE.finditer(log)])
    rews  = np.array([float(m.group(1)) for m in _REW_RE.finditer(log)])
    return steps, rews


def smooth(y, window=50):
    if len(y) <= window:
        return y, np.arange(len(y))
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid'), np.arange(window - 1, len(y))


# ════════════════════════════════════════════════════════════════════════════
# 1. ИГРОВАЯ ИЛЛЮСТРАЦИЯ — борд + 3 фигуры снизу
# ════════════════════════════════════════════════════════════════════════════
GAME_BOARD = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0],
])

# Три фигуры для выбора
PIECES = [
    np.array([[1, 1], [1, 1]]),       # 2×2 квадрат
    np.array([[1, 0], [1, 1]]),       # L-тримино
    np.array([[1, 1, 1]]),            # I-3 горизонталь
]
PIECE_COLORS  = [BLUE_BLOB, ORANGE, GREEN_HEAT]
PIECE_BORDERS = ["#6ab4ff", "#ffaa55", "#4ade80"]
PIECE_LABELS  = ["2×2", "L-trim.", "I-3"]

fig = plt.figure(figsize=(5.4, 8.0))
fig.patch.set_facecolor(BG)

# Борд занимает верхние 8 единиц
ax = fig.add_axes([0.02, 0.30, 0.96, 0.68])
ax.set_facecolor(BG)

for r in range(8):
    for c in range(8):
        draw_cell(ax, c, 7 - r, CELL_FILLED if GAME_BOARD[r, c] else CELL_EMPTY)

ax.set_xlim(-0.1, 8.1); ax.set_ylim(-0.2, 8.1)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Состояние доски", fontsize=12, color=MUTED, pad=4)

# Разделительная полоса
ax_div = fig.add_axes([0.05, 0.27, 0.90, 0.01])
ax_div.set_facecolor(GRID_STROKE); ax_div.axis("off")

# Три фигуры снизу
ax_p = fig.add_axes([0.02, 0.01, 0.96, 0.25])
ax_p.set_facecolor(BG); ax_p.axis("off")
ax_p.set_xlim(0, 12); ax_p.set_ylim(-0.5, 3.5)
ax_p.set_title("Доступные фигуры (выбери одну из трёх)", fontsize=11,
               color=MUTED, pad=2)

positions = [0.5, 4.5, 8.5]   # x-начало для каждой фигуры
for piece, color, border, label, px in zip(
        PIECES, PIECE_COLORS, PIECE_BORDERS, PIECE_LABELS, positions):
    h, w = piece.shape
    # Центрируем фигуру в окне шириной ~3 ячейки
    off_c = (3 - w) / 2
    off_r = (3 - h) / 2
    for r in range(h):
        for c in range(w):
            if piece[r, c]:
                draw_cell(ax_p, px + off_c + c, off_r + (h - 1 - r),
                          color, border=border, lw=1.2, alpha=0.90)
    ax_p.text(px + 1.5, -0.35, label, ha="center", va="top",
              fontsize=10, color=color, fontweight="bold")

_demo_path = os.path.join(CHARTS_DIR, "board_game_demo.png")
plt.savefig(_demo_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
plt.close()
shutil.copy2(_demo_path, os.path.join(PICTURES_DIR, "board_game_demo.png"))
print("  ✓ results/charts/board_game_demo.png → presentation/tex/pictures/board_game_demo.png")


# ════════════════════════════════════════════════════════════════════════════
# 3. СРАВНЕНИЕ ЛИНИЙ (основной результат)
# ════════════════════════════════════════════════════════════════════════════
agents = ["Random", "Heuristic", "MLP\nGen 1", "MLP\nGen 2", "MLP\nGen 3"]
means  = [1.79,     8.48,        8.84,           9.88,          11.21]
stds   = [2.12,     7.60,        5.90,           6.57,           7.68]
colors = [GREY,     ORANGE,      BLUE_BLOB,      GREEN_HEAT,     PURPLE]

fig, ax = plt.subplots(figsize=(10, 5.0))
bars = ax.bar(agents, means, color=colors, width=0.55, alpha=0.82, zorder=3,
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

ax.axhline(8.48, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=2)
ax.text(4.3, 8.9, "Heuristic baseline", fontsize=10, color=ORANGE, alpha=0.7)
plt.tight_layout()
save("chart_lines.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. КРИВЫЕ ОБУЧЕНИЯ — MLP
# ════════════════════════════════════════════════════════════════════════════
MLP_MODELS = ["MLP Gen 1", "MLP Gen 2", "MLP Gen 3"]
MLP_COLORS = [BLUE_BLOB, GREEN_HEAT, PURPLE]

fig, ax = plt.subplots(figsize=(11, 5.2))

for name, color in zip(MLP_MODELS, MLP_COLORS):
    steps, rews = parse_log(LOG_PATHS[name])
    steps_m = steps / 1e6  # в миллионах
    win = 80 if len(rews) > 1000 else 40

    # Сырые данные — лёгкая полоса
    ax.plot(steps_m, rews, color=color, alpha=0.12, linewidth=0.8, zorder=2)

    # Сглаженная кривая
    s_rews, s_idx = smooth(rews, win)
    ax.plot(steps_m[s_idx], s_rews, color=color, linewidth=2.2,
            label=name, zorder=3)

    # Финальное значение
    final = s_rews[-1]
    ax.text(steps_m[s_idx[-1]] + 0.04, final,
            f"{final:.1f}", va="center", fontsize=10, color=color,
            fontweight="bold")

ax.axhline(36.1, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)
ax.text(0.05, 37.0, "Heuristic (≈36)", fontsize=10, color=ORANGE, alpha=0.7)

ax.set_xlabel("Шаги обучения (млн)", fontsize=13, labelpad=8)
ax.set_ylabel("ep_rew_mean (скользящее среднее)", fontsize=13, labelpad=8)
ax.set_title("Кривые обучения — MLP агенты", fontsize=14, pad=12)
ax.set_xlim(0, 5.2)
ax.yaxis.grid(True, alpha=0.20, zorder=0); ax.set_axisbelow(True)
ax.legend(fontsize=12, framealpha=0.2, loc="upper left")
plt.tight_layout()
save("chart_learning_mlp.png")


# ════════════════════════════════════════════════════════════════════════════
# 6. КРИВЫЕ ОБУЧЕНИЯ — CNN
# ════════════════════════════════════════════════════════════════════════════
CNN_MODELS = ["CNN Gen 1", "CNN Gen 2", "CNN Gen 3", "CNN Gen 3 TR"]
CNN_COLORS = [TEAL, RED_DEAD, PURPLE, YELLOW_HIGH]

fig, ax = plt.subplots(figsize=(11, 5.2))

for name, color in zip(CNN_MODELS, CNN_COLORS):
    steps, rews = parse_log(LOG_PATHS[name])
    steps_m = steps / 1e6
    win = 80 if len(rews) > 1000 else 40

    ax.plot(steps_m, rews, color=color, alpha=0.12, linewidth=0.8, zorder=2)

    s_rews, s_idx = smooth(rews, win)
    ax.plot(steps_m[s_idx], s_rews, color=color, linewidth=2.2,
            label=name, zorder=3)

    final = s_rews[-1]
    ax.text(steps_m[s_idx[-1]] + 0.04, final,
            f"{final:.1f}", va="center", fontsize=10, color=color,
            fontweight="bold")

ax.axhline(36.1, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)
ax.text(0.05, 37.0, "Heuristic (≈36)", fontsize=10, color=ORANGE, alpha=0.7)

ax.set_xlabel("Шаги обучения (млн)", fontsize=13, labelpad=8)
ax.set_ylabel("ep_rew_mean (скользящее среднее)", fontsize=13, labelpad=8)
ax.set_title("Кривые обучения — CNN агенты", fontsize=14, pad=12)
ax.set_xlim(0, 5.2)
ax.yaxis.grid(True, alpha=0.20, zorder=0); ax.set_axisbelow(True)
ax.legend(fontsize=12, framealpha=0.2, loc="upper left")
plt.tight_layout()
save("chart_learning_cnn.png")


# ════════════════════════════════════════════════════════════════════════════
# ДАННЫЕ — статические метрики всех агентов
# ════════════════════════════════════════════════════════════════════════════
AGENT_LINES = {
    "Random":       {"mean": 1.79,  "std": 2.12},
    "Heuristic":    {"mean": 8.48,  "std": 7.60},
    "MLP Gen 1":    {"mean": 8.84,  "std": 5.90},
    "MLP Gen 2":    {"mean": 9.88,  "std": 6.57},
    "MLP Gen 3":    {"mean": 11.21, "std": 7.68},
    "CNN Gen 1":    {"mean": 7.36,  "std": 4.74},
    "CNN Gen 2":    {"mean": 8.50,  "std": 5.72},
    "CNN Gen 3":    {"mean": 4.73,  "std": 3.79},
    "CNN Gen 3 TR": {"mean": 8.85,  "std": 5.66},
}


def _has_data(name: str) -> bool:
    return AGENT_LINES.get(name, {}).get("mean") is not None

THRESHOLDS = [1, 5, 10, 15, 20, 25, 30]
SURVIVAL = {
    "Random":       {1: 0.66, 5: 0.12, 10: 0.02, 15: 0.00, 20: 0.00, 25: 0.00, 30: 0.00},
    "Heuristic":    {1: 0.94, 5: 0.64, 10: 0.33, 15: 0.17, 20: 0.09, 25: 0.04, 30: 0.03},
    "MLP Gen 1":    {1: 0.80, 5: 0.62, 10: 0.35, 15: 0.14, 20: 0.04, 25: 0.01, 30: 0.00},
    "MLP Gen 2":    {1: 0.84, 5: 0.69, 10: 0.45, 15: 0.25, 20: 0.12, 25: 0.05, 30: 0.02},
    "MLP Gen 3":    {1: 0.99, 5: 0.85, 10: 0.49, 15: 0.26, 20: 0.14, 25: 0.07, 30: 0.03},
    "CNN Gen 1":    {1: 0.75, 5: 0.55, 10: 0.28, 15: 0.11, 20: 0.03, 25: 0.01, 30: 0.00},
    "CNN Gen 2":    {1: 0.83, 5: 0.67, 10: 0.43, 15: 0.23, 20: 0.10, 25: 0.04, 30: 0.01},
    "CNN Gen 3":    None,
    "CNN Gen 3 TR": {1: 0.99, 5: 0.78, 10: 0.40, 15: 0.15, 20: 0.06, 25: 0.02, 30: 0.01},
}

AGENT_COLORS = {
    "Random":       GREY,
    "Heuristic":    ORANGE,
    "MLP Gen 1":    BLUE_BLOB,
    "MLP Gen 2":    GREEN_HEAT,
    "MLP Gen 3":    PURPLE,
    "CNN Gen 1":    TEAL,
    "CNN Gen 2":    RED_DEAD,
    "CNN Gen 3":    MUTED,
    "CNN Gen 3 TR": YELLOW_HIGH,
}

ALL_AGENTS = ["Random", "Heuristic",
              "MLP Gen 1", "MLP Gen 2", "MLP Gen 3",
              "CNN Gen 1", "CNN Gen 2", "CNN Gen 3", "CNN Gen 3 TR"]


# ════════════════════════════════════════════════════════════════════════════
# 7. ВЫЖИВАЕМОСТЬ — ВСЕ 8 АГЕНТОВ
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5.5))

xs = [0] + THRESHOLDS
_survival_all_agents = [n for n in ALL_AGENTS if n != "CNN Gen 3"]
for name in _survival_all_agents:
    color = AGENT_COLORS[name]
    if SURVIVAL[name] is None:
        ax.plot(xs, [0] * len(xs), color=color, linewidth=1.2,
                linestyle=":", alpha=0.5, label=f"{name} (нет данных)", zorder=1)
        continue

    rates = SURVIVAL[name]
    ys = [1.0] + [rates[t] for t in THRESHOLDS]

    if name in ("Random", "Heuristic"):
        lw, ls, z = 1.8, "--", 2
    elif "MLP" in name:
        lw, ls, z = 2.4, "-", 3
    else:
        lw, ls, z = 2.0, "-.", 3

    ax.step(xs, ys, where="post", color=color, linewidth=lw,
            linestyle=ls, alpha=0.90, label=name, zorder=z)
    ax.plot(THRESHOLDS, [rates[t] for t in THRESHOLDS],
            "o", color=color, markersize=4, alpha=0.65, zorder=z + 1)

ax.set_xlabel("Число очищенных линий (порог)", fontsize=13, labelpad=8)
ax.set_ylabel("Доля эпизодов ≥ threshold", fontsize=13, labelpad=8)
ax.set_title("Survival Analysis — все агенты", fontsize=14, pad=12)
ax.set_xlim(0, 32); ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.grid(True, alpha=0.20, zorder=0)
ax.xaxis.grid(True, alpha=0.12, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.2, loc="upper right",
          ncol=2, columnspacing=1.0)
plt.tight_layout()
save("chart_survival_all.png")


# ════════════════════════════════════════════════════════════════════════════
# 8. СТАБИЛЬНОСТЬ (CV) — ВСЕ 8 АГЕНТОВ
# ════════════════════════════════════════════════════════════════════════════
# CV = std / mean
CV_DATA = {}
for name in ALL_AGENTS:
    d = AGENT_LINES[name]
    CV_DATA[name] = d["std"] / d["mean"] if d["mean"] else None

fig, ax = plt.subplots(figsize=(10, 5.5))

y_pos = np.arange(len(ALL_AGENTS))
bar_colors = [AGENT_COLORS[n] for n in ALL_AGENTS]
cv_vals = [CV_DATA[n] if CV_DATA[n] is not None else 0.0 for n in ALL_AGENTS]

bars = ax.barh(y_pos, cv_vals, color=bar_colors, alpha=0.85,
               height=0.55, zorder=3)

# Штриховка для агентов без данных
for i, name in enumerate(ALL_AGENTS):
    if CV_DATA[name] is None:
        bars[i].set_hatch("///")
        bars[i].set_alpha(0.30)

for i, (name, cv) in enumerate(zip(ALL_AGENTS, cv_vals)):
    if CV_DATA[name] is None:
        ax.text(0.04, i, "нет данных", va="center", fontsize=10,
                color=MUTED, style="italic")
    else:
        ax.text(cv + 0.015, i, f"{cv:.2f}",
                va="center", fontsize=11, color=WHITE, fontweight="bold")

# Справочные линии
for ref, label_text, col in [
    (0.70, "← нейросети", GREEN_HEAT),
    (0.91, "Heuristic", ORANGE),
]:
    ax.axvline(ref, color=col, linestyle=":", linewidth=1.2, alpha=0.45, zorder=4)

ax.text(0.72, 7.5, "← нейросети", fontsize=9, color=GREEN_HEAT, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(ALL_AGENTS, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("Коэффициент вариации CV = σ / μ  (меньше = стабильнее)",
              fontsize=12, labelpad=8)
ax.set_title("Стабильность — все агенты", fontsize=14, pad=12)
ax.set_xlim(0, 1.45)
ax.xaxis.grid(True, alpha=0.20, zorder=0); ax.set_axisbelow(True)
plt.tight_layout()
save("chart_stability_all.png")


# ════════════════════════════════════════════════════════════════════════════
# 9–15. СТАТИСТИЧЕСКИЕ ТЕСТЫ
# ════════════════════════════════════════════════════════════════════════════

MW_RESULTS = {
    "MLP Gen 1":    (0.534, 0.141),
    "MLP Gen 2":    (0.613, 0.0028),
    "MLP Gen 3":    (0.634, 0.0000),
    "CNN Gen 1":    (0.503, 0.820),
    "CNN Gen 2":    (0.591, 0.011),
    "CNN Gen 3 TR": (0.571, 0.0000),
}
COHENS_D = {
    "MLP Gen 1":    +0.07,
    "MLP Gen 2":    +0.37,
    "MLP Gen 3":    +0.348,
    "CNN Gen 1":    -0.10,
    "CNN Gen 2":    +0.30,
    "CNN Gen 3 TR": +0.078,
}
BOOTSTRAP_CI = {
    "MLP Gen 1":    (+0.43,  -0.72, +1.58),
    "MLP Gen 2":    (+2.57,  +1.31, +3.84),
    "MLP Gen 3":    (+2.70,  +2.02, +3.37),
    "CNN Gen 1":    (-0.69,  -1.93, +0.55),
    "CNN Gen 2":    (+2.01,  +0.78, +3.24),
    "CNN Gen 3 TR": (+0.54,  -0.07, +1.16),
}
WILCOXON = {
    "MLP Gen 1":    (0.530, 0.092,  +0.5),
    "MLP Gen 2":    (0.600, 0.0012, +2.0),
    "MLP Gen 3":    (0.594, 0.0000, +2.0),
    "CNN Gen 1":    (0.490, 0.620,  -0.5),
    "CNN Gen 2":    (0.578, 0.015,  +1.5),
    "CNN Gen 3 TR": (0.547, 0.0010, +1.0),
}

mlp_agents = ["MLP Gen 1", "MLP Gen 2", "MLP Gen 3"]
cnn_agents = ["CNN Gen 1", "CNN Gen 2", "CNN Gen 3 TR"]


# ── 9. Mann-Whitney CLES ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("Mann–Whitney U: CLES  (P(агент > Heuristic) | 0.5 = паритет)",
             fontsize=14, color=WHITE, y=1.01)

for ax, group, title in zip(axes, [mlp_agents, cnn_agents],
                             ["MLP агенты", "CNN агенты"]):
    cles_vals, p_vals, clrs, labels = [], [], [], []
    for name in group:
        if name not in MW_RESULTS:
            cles_vals.append(0.0); p_vals.append(None)
            clrs.append(AGENT_COLORS[name]); labels.append(name)
        else:
            c, p = MW_RESULTS[name]
            cles_vals.append(c); p_vals.append(p)
            clrs.append(AGENT_COLORS[name]); labels.append(name)

    y = np.arange(len(labels))
    ax.barh(y, cles_vals, color=clrs, alpha=0.85, height=0.50, zorder=3)
    ax.axvline(0.5, color=WHITE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=4)

    for i, (name, p) in enumerate(zip(labels, p_vals)):
        val = cles_vals[i]
        if p is None:
            ax.text(0.26, i, "нет данных", va="center", ha="center",
                    fontsize=11, color=MUTED, style="italic")
        else:
            ax.text(val + 0.008, i, sig_label(p),
                    va="center", fontsize=11, color=WHITE, fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0.35, 0.75); ax.set_xlabel("CLES", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.xaxis.grid(True, alpha=0.20, zorder=0); ax.set_axisbelow(True)
    ax.text(0.355, -0.7, "ns = p≥0.05  * = p<0.05  ** = p<0.01  *** = p<0.001",
            fontsize=8, color=MUTED)

plt.tight_layout()
save("chart_stat_mann_whitney.png")


# ── 10. Cohen's d ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("Cohen's d — размер эффекта (агент − Heuristic, метрика: lines)",
             fontsize=14, color=WHITE, y=1.01)

for ax, group, title in zip(axes, [mlp_agents, cnn_agents],
                             ["MLP агенты", "CNN агенты"]):
    d_vals, clrs, labels = [], [], []
    for name in group:
        if name not in COHENS_D:
            d_vals.append(0.0); clrs.append(AGENT_COLORS[name]); labels.append(name)
        else:
            d_vals.append(COHENS_D[name])
            clrs.append(AGENT_COLORS[name]); labels.append(name)

    y = np.arange(len(labels))
    ax.barh(y, d_vals, color=clrs, alpha=0.85, height=0.50, zorder=3)
    ax.axvline(0.0, color=WHITE, linestyle="-", linewidth=1.0, alpha=0.5, zorder=4)

    for ref, lbl, col in [(0.2, "мало", YELLOW_HIGH),
                           (0.5, "средне", ORANGE),
                           (0.8, "много", RED_DEAD)]:
        ax.axvline(ref, color=col, linestyle=":", linewidth=1.0, alpha=0.45, zorder=3)
        ax.text(ref + 0.01, len(labels) - 0.1, lbl,
                fontsize=8, color=col, alpha=0.7, va="top")

    for i, (name, d) in enumerate(zip(labels, d_vals)):
        if name not in COHENS_D:
            ax.text(0.02, i, "нет данных", va="center", ha="left",
                    fontsize=11, color=MUTED, style="italic")
        else:
            off = 0.015 if d >= 0 else -0.08
            ax.text(d + off, i, f"{d:+.2f}",
                    va="center", fontsize=11, color=WHITE, fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(-0.30, 0.65); ax.set_xlabel("Cohen's d", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.xaxis.grid(True, alpha=0.20, zorder=0); ax.set_axisbelow(True)

plt.tight_layout()
save("chart_stat_cohens_d.png")


# ── 11. Bootstrap 95% CI ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("Bootstrap 95% CI разницы средних (агент − Heuristic, lines)",
             fontsize=14, color=WHITE, y=1.01)

for ax, group, title in zip(axes, [mlp_agents, cnn_agents],
                             ["MLP агенты", "CNN агенты"]):
    y = np.arange(len(group))
    labels = []
    for i, name in enumerate(group):
        if name not in BOOTSTRAP_CI:
            labels.append(name)
            ax.text(0.0, i, "нет данных", va="center", ha="center",
                    fontsize=11, color=MUTED, style="italic")
            continue

        obs, lo, hi = BOOTSTRAP_CI[name]
        color = AGENT_COLORS[name]
        is_sig = (lo > 0) or (hi < 0)
        ax.plot([lo, hi], [i, i], color=color,
                linewidth=2.5 if is_sig else 1.5,
                alpha=0.9 if is_sig else 0.55, zorder=3)
        ax.plot(obs, i, "o", color=color, markersize=9, zorder=4)
        ax.plot([lo, hi], [i, i], "|", color=color,
                markersize=10, markeredgewidth=2, zorder=4)
        suffix = "  ✓" if is_sig else "  ✗"
        ax.text(hi + 0.15, i,
                f"{obs:+.2f} [{lo:+.1f}, {hi:+.1f}]{suffix}",
                va="center", fontsize=10, color=WHITE)
        labels.append(name)

    ax.axvline(0.0, color=WHITE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(-3.5, 8.5); ax.set_xlabel("Разница средних (lines)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.xaxis.grid(True, alpha=0.20, zorder=0); ax.set_axisbelow(True)
    ax.text(-3.4, -0.7, "✓ CI не включает 0  ✗ CI включает 0",
            fontsize=8, color=MUTED)

plt.tight_layout()
save("chart_stat_bootstrap_ci.png")


# ── 12. Wilcoxon paired win rate ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("Wilcoxon paired: Win Rate vs Heuristic (одинаковые эпизоды)",
             fontsize=14, color=WHITE, y=1.01)

for ax, group, title in zip(axes, [mlp_agents, cnn_agents],
                             ["MLP агенты", "CNN агенты"]):
    wr_vals, p_vals, med_diffs, clrs, labels = [], [], [], [], []
    for name in group:
        if name not in WILCOXON:
            wr_vals.append(0.0); p_vals.append(None)
            med_diffs.append(None); clrs.append(AGENT_COLORS[name]); labels.append(name)
        else:
            wr, p, md = WILCOXON[name]
            wr_vals.append(wr); p_vals.append(p)
            med_diffs.append(md); clrs.append(AGENT_COLORS[name])
            labels.append(name)

    y = np.arange(len(labels))
    ax.barh(y, wr_vals, color=clrs, alpha=0.85, height=0.50, zorder=3)
    ax.axvline(0.5, color=WHITE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=4)

    for i, (name, wr, p, md) in enumerate(zip(labels, wr_vals, p_vals, med_diffs)):
        if p is None:
            ax.text(0.26, i, "нет данных", va="center", ha="center",
                    fontsize=11, color=MUTED, style="italic")
        else:
            ax.text(0.36, i + 0.30,
                    f"{wr:.1%}  {sig_label(p)}  (Δmed={md:+.1f})",
                    va="bottom", fontsize=9, color=WHITE, fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0.35, 0.80)
    ax.set_xlabel("Win Rate (агент > Heuristic)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=8)
    ax.xaxis.grid(True, alpha=0.20, zorder=0); ax.set_axisbelow(True)
    ax.text(0.355, -0.7, "ns = p≥0.05  * = p<0.05  ** = p<0.01  *** = p<0.001",
            fontsize=8, color=MUTED)

plt.tight_layout()
save("chart_stat_wilcoxon.png")


# ── 13. Survival curves (детальные — две панели) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Survival Analysis — P(lines ≥ threshold)",
             fontsize=14, color=WHITE, y=1.01)

curve_groups = [
    ("MLP агенты", ["Heuristic", "MLP Gen 1", "MLP Gen 2", "MLP Gen 3"]),
    ("CNN агенты", ["Heuristic", "CNN Gen 1", "CNN Gen 2", "CNN Gen 3 TR"]),
]
xs = [0] + THRESHOLDS
for ax, (title, group) in zip(axes, curve_groups):
    for name in group:
        color = AGENT_COLORS[name]
        lw = 2.5 if name != "Heuristic" else 1.8
        ls = "--" if name == "Heuristic" else "-"
        if SURVIVAL[name] is None:
            ax.plot(xs, [0] * len(xs), color=color, linewidth=1.2,
                    linestyle=":", alpha=0.5, label=f"{name} (нет данных)", zorder=1)
            continue
        rates = SURVIVAL[name]
        ys = [1.0] + [rates[t] for t in THRESHOLDS]
        ax.step(xs, ys, where="post", color=color, linewidth=lw,
                linestyle=ls, alpha=0.90, label=name, zorder=3)
        ax.plot(THRESHOLDS, [rates[t] for t in THRESHOLDS],
                "o", color=color, markersize=5, alpha=0.7, zorder=4)

    ax.set_xlabel("Порог (lines)", fontsize=12)
    ax.set_ylabel("P(lines ≥ threshold)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlim(0, 32); ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.grid(True, alpha=0.20, zorder=0)
    ax.xaxis.grid(True, alpha=0.12, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.2, loc="upper right")

plt.tight_layout()
save("chart_stat_survival_curves.png")


# ── 14–15. Сводные таблицы ───────────────────────────────────────────────────
def draw_stat_table(agents_list, title, filename):
    col_labels = ["Агент", "Mean±σ (lines)", "CLES",
                  "p (MW)", "Cohen's d", "Boot CI 95%",
                  "Win Rate", "p (Wilcox)"]
    rows = []
    for name in agents_list:
        if name not in MW_RESULTS:
            m = AGENT_LINES[name]
            mean_std = f"{m['mean']:.2f}±{m['std']:.2f}" if m["mean"] else "—"
            rows.append([name, mean_std] + ["—"] * (len(col_labels) - 2))
            continue
        m = AGENT_LINES[name]
        cles, p_mw = MW_RESULTS[name]
        d = COHENS_D[name]
        obs, lo, hi = BOOTSTRAP_CI[name]
        wr, p_wil, _ = WILCOXON[name]
        rows.append([
            name,
            f"{m['mean']:.2f}±{m['std']:.2f}",
            f"{cles:.3f}",
            f"{p_mw:.4f} {sig_label(p_mw)}",
            f"{d:+.2f}",
            f"[{lo:+.1f},{hi:+.1f}]",
            f"{wr:.1%}",
            f"{p_wil:.4f} {sig_label(p_wil)}",
        ])

    n_rows, n_cols = len(rows), len(col_labels)
    fig, ax = plt.subplots(figsize=(16, 1.0 + n_rows * 0.55))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, color=WHITE, y=0.99)

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_STROKE)
        if row == 0:
            cell.set_facecolor(GRID_STROKE)
            cell.set_text_props(color=WHITE, fontweight="bold")
        elif col == 0:
            name = agents_list[row - 1]
            cell.set_facecolor(AGENT_COLORS.get(name, MUTED))
            cell.set_text_props(color=WHITE, fontweight="bold")
        else:
            bg = "#161630" if row % 2 == 0 else CARD
            cell.set_facecolor(bg)
            content = cell.get_text().get_text()
            col_c = WHITE
            if "***" in content or "**" in content:
                cell.set_facecolor("#1a3020"); col_c = "#6eff9a"
            elif content == "—":
                col_c = MUTED
            cell.set_text_props(color=col_c)

    ax.text(0.5, -0.04,
            "vs Heuristic (baseline). n=500 эпизодов. "
            "MW = Mann–Whitney U. Wilcox = Wilcoxon signed-rank (paired).",
            ha="center", va="top", fontsize=8, color=MUTED,
            transform=ax.transAxes)

    plt.tight_layout(pad=0.5)
    save(filename)


draw_stat_table(mlp_agents,
                "Статистические тесты — MLP агенты vs Heuristic (lines)",
                "chart_stat_table_mlp.png")

draw_stat_table(cnn_agents,
                "Статистические тесты — CNN агенты vs Heuristic (lines)",
                "chart_stat_table_cnn.png")


# ── 16. MLP vs CNN сравнение ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.0))

# Gen 1, Gen 2, Gen 3 (CNN = CNN Gen 3 TR)
gen_labels   = ["Gen 1", "Gen 2", "Gen 3"]
mlp_means    = [8.84,  9.88,  11.21]
mlp_stds     = [5.90,  6.57,   7.68]
cnn_means    = [7.36,  8.50,   8.85]
cnn_stds     = [4.74,  5.72,   5.66]
x = np.arange(len(gen_labels)); w = 0.35

mlp_plot = [v if v else 0 for v in mlp_means]
b_mlp = ax.bar(x - w/2, mlp_plot, width=w, color=BLUE_BLOB, alpha=0.85,
               label="MLP", zorder=3,
               yerr=[s if s else 0 for s in mlp_stds], capsize=6,
               error_kw={"color": WHITE, "linewidth": 1.4, "alpha": 0.5})
b_cnn = ax.bar(x + w/2, cnn_means, width=w, color=RED_DEAD, alpha=0.85,
               label="CNN", zorder=3,
               yerr=cnn_stds, capsize=6,
               error_kw={"color": WHITE, "linewidth": 1.4, "alpha": 0.5})

for bar, val in zip(b_mlp, mlp_means):
    if val is not None:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, color=WHITE)

for bar, val in zip(b_cnn, cnn_means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=11, color=WHITE)

ax.axhline(8.48, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=2)
ax.text(2.2, 8.7, "Heuristic (8.48)", fontsize=9, color=ORANGE, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(gen_labels, fontsize=12)
ax.set_ylabel("Линий за игру (среднее ± σ)", fontsize=13, labelpad=10)
ax.set_title("MLP vs CNN — сравнение по поколениям\n(CNN Gen 3 = CNN Gen 3 TR)", fontsize=13, pad=12)
ax.set_ylim(0, 22)
ax.yaxis.grid(True, alpha=0.25, zorder=0); ax.set_axisbelow(True)
ax.legend(fontsize=12, framealpha=0.2)
plt.tight_layout()
save("chart_mlp_vs_cnn.png")


# ════════════════════════════════════════════════════════════════════════════
# 17. КРИВЫЕ ОБУЧЕНИЯ — Gen 4 (CNN + ViT, 16×16, BC+PPO)
# ════════════════════════════════════════════════════════════════════════════
GEN4_MODELS = ["CNN Gen 4", "ViT Gen 4"]
GEN4_COLORS = [TEAL, PURPLE]

fig, ax = plt.subplots(figsize=(11, 5.2))

any_plotted = False
for name, color in zip(GEN4_MODELS, GEN4_COLORS):
    path = LOG_PATHS[name]
    if not os.path.exists(path):
        print(f"  ⚠ лог не найден: {path}")
        continue
    steps, rews = parse_log(path)
    if len(rews) == 0:
        print(f"  ⚠ лог пустой: {path}")
        continue
    steps_m = steps / 1e6
    win = 80 if len(rews) > 1000 else 20

    ax.plot(steps_m, rews, color=color, alpha=0.12, linewidth=0.8, zorder=2)

    s_rews, s_idx = smooth(rews, win)
    ax.plot(steps_m[s_idx], s_rews, color=color, linewidth=2.2,
            label=name, zorder=3)

    final = s_rews[-1]
    ax.text(steps_m[s_idx[-1]] + 0.1, final,
            f"{final:.1f}", va="center", fontsize=10, color=color,
            fontweight="bold")
    any_plotted = True

if any_plotted:
    ax.axhline(64.23, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.5, zorder=1)
    ax.text(0.1, 66, "Heuristic 16×16 (≈64)", fontsize=10, color=ORANGE, alpha=0.7)

    ax.set_xlabel("Шаги обучения (млн)", fontsize=13, labelpad=8)
    ax.set_ylabel("ep_rew_mean (скользящее среднее)", fontsize=13, labelpad=8)
    ax.set_title("Кривые обучения — Gen 4 (CNN и ViT, 16×16, BC+PPO)",
                 fontsize=14, pad=12)
    ax.yaxis.grid(True, alpha=0.20, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, framealpha=0.2, loc="lower right")
    plt.tight_layout()
    save("chart_learning_gen4.png")
else:
    plt.close()
    print("  ⚠ chart_learning_gen4.png пропущен — логи не найдены")


# ════════════════════════════════════════════════════════════════════════════
# 18. СРАВНЕНИЕ АГЕНТОВ GEN 4 — доска 16×16
# ════════════════════════════════════════════════════════════════════════════
agents4   = ["Random", "Heuristic", "CNN\n(BC+PPO)", "ViT\n(BC+PPO)"]
means4    = [0.47,      7.21,        18.76,            18.12]
stds4     = [0.89,      7.09,         9.96,             9.74]
colors4   = [GREY,      ORANGE,       TEAL,             PURPLE]

fig, ax = plt.subplots(figsize=(9, 5.0))
bars = ax.bar(agents4, means4, color=colors4, width=0.55, alpha=0.82, zorder=3,
              yerr=stds4, capsize=7,
              error_kw={"color": WHITE, "linewidth": 1.6, "alpha": 0.6})
ax.set_ylabel("Линий за игру (среднее ± σ)", fontsize=13, labelpad=10)
ax.set_title("Результаты Gen 4 — доска 16×16 (500 эпизодов, seed=42)",
             fontsize=14, pad=14, color=WHITE)
ax.set_ylim(0, 35)
ax.yaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

for bar, mean in zip(bars, means4):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.6,
            f"{mean:.2f}", ha="center", va="bottom",
            fontsize=12, color=WHITE, fontweight="bold")

ax.axhline(7.21, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.4, zorder=2)
ax.text(3.3, 7.9, "Heuristic baseline", fontsize=10, color=ORANGE, alpha=0.7)


plt.tight_layout()
save("chart_comparison_gen4.png")


# ════════════════════════════════════════════════════════════════════════════
# 19. SURVIVAL ANALYSIS GEN 4
# ════════════════════════════════════════════════════════════════════════════
THRESHOLDS_G4 = [1, 5, 10, 15, 20, 25, 30]
SURVIVAL_GEN4 = {
    "Random":       {1: 0.29, 5: 0.01, 10: 0.00, 15: 0.00, 20: 0.00, 25: 0.00, 30: 0.00},
    "Heuristic":    {1: 0.88, 5: 0.51, 10: 0.32, 15: 0.15, 20: 0.07, 25: 0.03, 30: 0.01},
    "CNN (BC+PPO)": {1: 1.00, 5: 0.97, 10: 0.85, 15: 0.63, 20: 0.41, 25: 0.27, 30: 0.13},
    "ViT (BC+PPO)": {1: 1.00, 5: 0.97, 10: 0.84, 15: 0.61, 20: 0.41, 25: 0.23, 30: 0.13},
}
SURV_G4_STYLES = [
    ("Random",       GREY,   ":",  1.4),
    ("Heuristic",    ORANGE, "--", 1.8),
    ("CNN (BC+PPO)", TEAL,   "-",  2.4),
    ("ViT (BC+PPO)", PURPLE, "-",  2.4),
]

fig, ax = plt.subplots(figsize=(10, 5.2))
xs4 = [0] + THRESHOLDS_G4
for name, color, ls, lw in SURV_G4_STYLES:
    rates = SURVIVAL_GEN4[name]
    ys = [1.0] + [rates[t] for t in THRESHOLDS_G4]
    ax.step(xs4, ys, where="post", color=color, linewidth=lw,
            linestyle=ls, alpha=0.90, label=name, zorder=3)
    ax.plot(THRESHOLDS_G4, [rates[t] for t in THRESHOLDS_G4],
            "o", color=color, markersize=5, alpha=0.7, zorder=4)

ax.set_xlabel("Порог (число очищенных линий)", fontsize=13, labelpad=8)
ax.set_ylabel("Доля эпизодов ≥ threshold", fontsize=13, labelpad=8)
ax.set_title("Survival Analysis Gen 4 — доска 16×16 (500 эпизодов)",
             fontsize=14, pad=12)
ax.set_xlim(0, 32); ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.grid(True, alpha=0.20, zorder=0)
ax.xaxis.grid(True, alpha=0.12, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=11, framealpha=0.2, loc="upper right")
plt.tight_layout()
save("chart_survival_gen4.png")


# ════════════════════════════════════════════════════════════════════════════
# 20. РАСШИРЕННЫЙ BARS — все поколения включая Gen 4
# ════════════════════════════════════════════════════════════════════════════
agents_all = ["Random", "Heuristic",
              "MLP\nGen 3", "CNN\nGen 3 TL",
              "CNN\n(BC+PPO)\n16×16", "ViT\n(BC+PPO)\n16×16"]
means_all  = [1.79,   8.48,   11.21,  8.85,   18.76,  18.12]
stds_all   = [2.12,   7.60,    7.68,  5.66,    9.96,   9.74]
colors_all = [GREY, ORANGE, PURPLE, YELLOW_HIGH, TEAL, BLUE_BLOB]

fig, ax = plt.subplots(figsize=(12, 5.2))
bars = ax.bar(agents_all, means_all, color=colors_all, width=0.60, alpha=0.82,
              zorder=3, yerr=stds_all, capsize=7,
              error_kw={"color": WHITE, "linewidth": 1.4, "alpha": 0.55})
ax.set_ylabel("Линий за игру (среднее ± σ)", fontsize=13, labelpad=10)
ax.set_title("Финальное сравнение: Gen 1–4 лучшие агенты",
             fontsize=14, pad=14, color=WHITE)
ax.set_ylim(0, 34)
ax.yaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

for bar, mean in zip(bars, means_all):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.5,
            f"{mean:.1f}", ha="center", va="bottom",
            fontsize=11, color=WHITE, fontweight="bold")

ax.axhline(8.48, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.35, zorder=2)
ax.text(5.3, 9.0, "Heuristic 8×8", fontsize=9, color=ORANGE, alpha=0.6)

# Разделитель 8×8 / 16×16
ax.axvline(3.5, color=GRID_STROKE, linestyle="-", linewidth=1.5, alpha=0.7, zorder=2)
ax.text(1.5, 30, "Доска 8×8", ha="center", fontsize=10, color=MUTED)
ax.text(4.5, 30, "Доска 16×16", ha="center", fontsize=10, color=MUTED)

plt.tight_layout()
save("chart_lines_final.png")


print("\nВсе графики сгенерированы → results/charts/")
