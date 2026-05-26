"""
generate_report_charts.py — академические графики для report.tex.

Запуск из корня проекта:
    python report/generate_report_charts.py

Выводит два файла в results/charts/report/:
    fig_learning_gen4.png   — кривые обучения Gen 4 (CNN + ViT)
    fig_comparison_gen4.png — столбчатая диаграмма: все агенты на 16×16
"""

import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ── Пути ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "results", "charts", "report")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_CNN = os.path.join(_ROOT, "models", "gen4", "cnn", "notebookc5318547d8.log")
LOG_VIT = os.path.join(_ROOT, "models", "gen4", "vit", "notebook7a01605639.log")

# ── Шрифт FreeSerif (тот же, что в report.tex) ───────────────────────────────
_FREESERIF_PATH = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
_FREESERIF_BOLD = "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"
if os.path.exists(_FREESERIF_PATH):
    fm.fontManager.addfont(_FREESERIF_PATH)
    fm.fontManager.addfont(_FREESERIF_BOLD)
    _FONT_FAMILY = "FreeSerif"
else:
    _FONT_FAMILY = "DejaVu Serif"

# ── Академический стиль ───────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#333333",
    "axes.linewidth":     0.8,
    "axes.labelcolor":    "black",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#cccccc",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    "text.color":         "black",
    "xtick.color":        "black",
    "ytick.color":        "black",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "font.family":        "serif",
    "font.serif":         [_FONT_FAMILY, "FreeSerif", "DejaVu Serif", "Times New Roman"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# Цветовая палитра (ColorBrewer, дружественная к дальтонизму)
C_CNN = "#2166ac"   # синий
C_VIT = "#d6604d"   # красно-оранжевый
C_HEU = "#555555"   # тёмно-серый
C_RND = "#aaaaaa"   # светло-серый

# ── Парсинг логов ─────────────────────────────────────────────────────────────
_STEPS_RE = re.compile(r'\|\s*total_timesteps\s*\|\s*([\d.e+]+)\s*\|')
_REW_RE   = re.compile(r'\|\s*ep_rew_mean\s*\|\s*([\d.e+\-]+)\s*\|')


def parse_log(path: str):
    text = open(path, encoding="utf-8", errors="replace").read()
    steps = [float(m.group(1)) for m in _STEPS_RE.finditer(text)]
    rews  = [float(m.group(1)) for m in _REW_RE.finditer(text)]
    n = min(len(steps), len(rews))
    return np.array(steps[:n]), np.array(rews[:n])


def smooth(y, window: int = 50):
    if len(y) <= window:
        return y, np.arange(len(y))
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid"), np.arange(window - 1, len(y))


def save(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  ✓ results/charts/report/{name}")


# ════════════════════════════════════════════════════════════════════════════
# 1. Кривые обучения Gen 4 (CNN + ViT)
# ════════════════════════════════════════════════════════════════════════════
HEURISTIC_REWARD_16 = 64.23

fig, ax = plt.subplots(figsize=(7.5, 4.0))

for log_path, label, color in [
    (LOG_CNN, "HierarchicalCNN", C_CNN),
    (LOG_VIT, "SmallViT",        C_VIT),
]:
    if not os.path.exists(log_path):
        print(f"  ⚠ лог не найден: {log_path}")
        continue
    steps, rews = parse_log(log_path)
    if len(rews) == 0:
        print(f"  ⚠ лог пустой: {log_path}")
        continue

    steps_m = steps / 1e6
    win = 60 if len(rews) > 300 else max(10, len(rews) // 5)

    ax.plot(steps_m, rews, color=color, alpha=0.15, linewidth=0.6, zorder=2)

    s_rews, s_idx = smooth(rews, win)
    ax.plot(steps_m[s_idx], s_rews, color=color, linewidth=1.8,
            label=label, zorder=3)

    final = s_rews[-1]
    ax.annotate(f"{final:.1f}", xy=(steps_m[s_idx[-1]], final),
                xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=9, color=color)

ax.axhline(HEURISTIC_REWARD_16, color=C_HEU, linestyle=":", linewidth=1.2,
           alpha=0.7, zorder=1,
           label=f"HeuristicAgent ({HEURISTIC_REWARD_16:.1f})")

ax.set_xlabel("Шаги обучения ($\\times 10^6$)")
ax.set_ylabel("Средняя награда за эпизод")
ax.set_title("Кривые обучения Gen-4 (PPO, доска $16\\times16$)")
ax.legend(loc="lower right")
ax.set_ylim(bottom=0)

plt.tight_layout()
save("fig_learning_gen4.png")


# ════════════════════════════════════════════════════════════════════════════
# 2. Сравнение агентов на 16×16
# ════════════════════════════════════════════════════════════════════════════
AGENTS = ["Random", "Heuristic", "CNN", "ViT"]
MEANS  = [0.47,      7.21,        18.76,  18.12]
STDS   = [0.89,      6.38,        11.43,  11.07]
COLORS = [C_RND,     C_HEU,       C_CNN,  C_VIT]

fig, ax = plt.subplots(figsize=(6.0, 4.0))

x = np.arange(len(AGENTS))
bars = ax.bar(x, MEANS, yerr=STDS, color=COLORS, width=0.5,
              capsize=5, error_kw={"linewidth": 1.0, "color": "#333333"},
              alpha=0.85, zorder=3)

for bar, mean, std in zip(bars, MEANS, STDS):
    ax.text(bar.get_x() + bar.get_width() / 2,
            mean + std + 0.5,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=9)

ax.axhline(7.21, color=C_HEU, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels(AGENTS, fontsize=10)
ax.set_ylabel("Среднее число очищенных линий ($\\pm\\sigma$)")
ax.set_title("Сравнение агентов на доске $16\\times16$\n($n = 500$ эпизодов)")
ax.set_ylim(0, 35)

plt.tight_layout()
save("fig_comparison_gen4.png")

# ════════════════════════════════════════════════════════════════════════════
# 3. Кривые обучения Gen 1–3 (CNN): порочный круг и трансфер
# ════════════════════════════════════════════════════════════════════════════
CNN_LOG_PATHS = {
    "CNN Gen 1":    os.path.join(_ROOT, "models", "gen1", "cnn_gen_1", "training_console.log"),
    "CNN Gen 2":    os.path.join(_ROOT, "models", "gen2", "cnn_gen_2", "training_console.log"),
    "CNN Gen 3":    os.path.join(_ROOT, "models", "gen3", "cnn_gen_3", "training_console.log"),
    "CNN Gen 3 TR": os.path.join(_ROOT, "models", "gen3", "cnn_gen_3_tr", "training_console.log"),
}

# Цвета по поколениям
C_GEN1 = "#4dac26"   # зелёный
C_GEN2 = "#b8860b"   # тёмно-жёлтый
C_GEN3 = "#aaaaaa"   # серый (провал)
C_G3TR = "#8b008b"   # фиолетовый (восстановление)

fig, ax = plt.subplots(figsize=(7.5, 4.0))

CNN_STYLES = [
    ("CNN Gen 1",    "CNN Gen~1",            C_GEN1, "-",  1.6),
    ("CNN Gen 2",    "CNN Gen~2",            C_GEN2, "-",  1.6),
    ("CNN Gen 3",    "CNN Gen~3 (с нуля)",   C_GEN3, "--", 1.4),
    ("CNN Gen 3 TR", "CNN Gen~3 (трансфер)", C_G3TR, "-",  1.8),
]

for key, label, color, ls, lw in CNN_STYLES:
    path = CNN_LOG_PATHS[key]
    if not os.path.exists(path):
        print(f"  ⚠ лог не найден: {path}")
        continue
    steps, rews = parse_log(path)
    if len(rews) == 0:
        print(f"  ⚠ лог пустой: {path}")
        continue

    steps_m = steps / 1e6
    win = 60 if len(rews) > 300 else max(10, len(rews) // 5)

    ax.plot(steps_m, rews, color=color, alpha=0.12, linewidth=0.6, zorder=2)

    s_rews, s_idx = smooth(rews, win)
    ax.plot(steps_m[s_idx], s_rews, color=color, linewidth=lw,
            linestyle=ls, label=label, zorder=3)

    final = s_rews[-1]
    ax.annotate(f"{final:.1f}", xy=(steps_m[s_idx[-1]], final),
                xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=9, color=color)

# Базовая линия эвристики (8×8)
HEURISTIC_REWARD_8 = 36.1
ax.axhline(HEURISTIC_REWARD_8, color=C_HEU, linestyle=":", linewidth=1.2,
           alpha=0.7, zorder=1, label=f"HeuristicAgent ({HEURISTIC_REWARD_8})")

# Аннотация: объяснение спада Gen 3
ax.annotate(
    "Порочный круг:\nтопологические каналы\nнеинформативны при\nслучайной политике",
    xy=(3.8, 33),           # стрелка указывает на спад
    xytext=(2.5, 12),       # текст ниже и левее
    fontsize=8,
    color=C_GEN3,
    arrowprops=dict(arrowstyle="->", color=C_GEN3, lw=1.0),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor=C_GEN3, linewidth=0.8, alpha=0.9),
)

ax.set_xlabel("Шаги обучения ($\\times 10^6$)")
ax.set_ylabel("Средняя награда за эпизод")
ax.set_title("Кривые обучения CNN Gen~1–3 (доска $8\\times8$)")
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(bottom=0)

plt.tight_layout()
save("fig_learning_cnn_gen123.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. Survival curves Gen 4
# ════════════════════════════════════════════════════════════════════════════
THRESHOLDS = [1, 5, 10, 15, 20, 25, 30]

SURVIVAL_GEN4 = {
    "Random":       {1: 0.29, 5: 0.01, 10: 0.00, 15: 0.00, 20: 0.00, 25: 0.00, 30: 0.00},
    "Heuristic":    {1: 0.88, 5: 0.51, 10: 0.32, 15: 0.15, 20: 0.07, 25: 0.03, 30: 0.01},
    "CNN (BC+PPO)": {1: 1.00, 5: 0.97, 10: 0.85, 15: 0.63, 20: 0.41, 25: 0.27, 30: 0.13},
    "ViT (BC+PPO)": {1: 1.00, 5: 0.97, 10: 0.84, 15: 0.61, 20: 0.41, 25: 0.23, 30: 0.13},
}

SURV_STYLES = [
    ("Random",       C_RND, ":",  1.2),
    ("Heuristic",    C_HEU, "--", 1.4),
    ("CNN (BC+PPO)", C_CNN, "-",  1.8),
    ("ViT (BC+PPO)", C_VIT, "-",  1.8),
]

fig, ax = plt.subplots(figsize=(7.0, 4.0))

xs = [0] + THRESHOLDS
for name, color, ls, lw in SURV_STYLES:
    rates = SURVIVAL_GEN4[name]
    ys = [1.0] + [rates[t] for t in THRESHOLDS]
    ax.step(xs, ys, where="post", color=color, linewidth=lw,
            linestyle=ls, label=name, zorder=3, alpha=0.9)
    ax.plot(THRESHOLDS, [rates[t] for t in THRESHOLDS],
            "o", color=color, markersize=4, alpha=0.7, zorder=4)

import matplotlib.ticker as mticker
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_xlabel("Порог числа очищенных линий")
ax.set_ylabel("Доля эпизодов, достигших порога")
ax.set_title("Survival Analysis Gen~4 ($16\\times16$, $n=500$ эпизодов)")
ax.set_xlim(0, 32)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
save("fig_survival_gen4.png")


print(f"\nГотово → results/charts/report/")
