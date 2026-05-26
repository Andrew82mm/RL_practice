"""
visualize_diagonal.py — визуализация состояния доски на заданном шаге игры.

Показывает борд CNN-агента в момент step N, где диагональное заполнение хорошо видно.

Запуск из корня проекта:
    python report/visualize_diagonal.py --step 83
    python report/visualize_diagonal.py --step 83 --episode 0
    python report/visualize_diagonal.py --step 83 --replay visualization/replay_cnn.json \
                                                  --replay-heu visualization/replay_heuristic.json
"""

import argparse
import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "results", "charts", "report")
os.makedirs(OUT_DIR, exist_ok=True)

_FREESERIF = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
if os.path.exists(_FREESERIF):
    fm.fontManager.addfont(_FREESERIF)
    _FONT = "FreeSerif"
else:
    _FONT = "DejaVu Serif"

mpl.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "text.color":        "black",
    "font.family":       "serif",
    "font.serif":        [_FONT, "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# Цвета
C_FILLED  = "#2166ac"   # занятая клетка
C_EMPTY   = "#f0f4f8"   # пустая клетка
C_LAST    = "#d6604d"   # последний поставленный ход
C_GRID    = "#cccccc"   # сетка


def draw_board(ax, board: np.ndarray, last_action=None, title: str = ""):
    """Рисует борд как сетку клеток."""
    n = board.shape[0]

    for r in range(n):
        for c in range(n):
            if board[r, c]:
                color = C_FILLED
            else:
                color = C_EMPTY
            rect = Rectangle((c, n - 1 - r), 1, 1,
                              facecolor=color, edgecolor=C_GRID, linewidth=0.4)
            ax.add_patch(rect)

    # Подсветить позицию последнего хода
    if last_action is not None:
        ay, ax_pos = last_action
        rect = Rectangle((ax_pos, n - 1 - ay), 1, 1,
                          facecolor=C_LAST, edgecolor="#8b0000",
                          linewidth=1.2, alpha=0.85)
        ax.add_patch(rect)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=6)


def get_frame(replay_path: str, episode: int, step: int):
    with open(replay_path) as f:
        d = json.load(f)
    episodes = d["episodes"]
    if episode >= len(episodes):
        raise ValueError(f"Эпизод {episode} не найден (всего {len(episodes)})")
    frames = episodes[episode]["frames"]
    if step >= len(frames):
        raise ValueError(f"Шаг {step} не найден (всего {len(frames)} шагов)")
    return frames[step], d["board_size"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=83,
                        help="Номер шага внутри эпизода (default: 83)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Индекс эпизода (default: 0)")
    parser.add_argument("--replay", default="visualization/replay_cnn.json",
                        help="Путь к replay CNN-агента")
    parser.add_argument("--replay-heu", default="visualization/replay_vit.json",
                        help="Путь к replay второго агента (default: ViT)")
    args = parser.parse_args()

    cnn_path = os.path.join(_ROOT, args.replay)
    heu_path = os.path.join(_ROOT, args.replay_heu)

    both_exist = os.path.exists(cnn_path) and os.path.exists(heu_path)

    if both_exist:
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    else:
        fig, axes_arr = plt.subplots(1, 1, figsize=(5, 5))
        axes = [axes_arr]

    configs = []
    if os.path.exists(cnn_path):
        configs.append((cnn_path, f"CNN (BC+PPO), шаг {args.step}"))
    if both_exist and os.path.exists(heu_path):
        configs.append((heu_path, f"ViT (BC+PPO), шаг {args.step}"))

    for ax, (path, title) in zip(axes if both_exist else [axes[0]], configs):
        try:
            frame, board_size = get_frame(path, args.episode, args.step)
        except ValueError as e:
            ax.set_title(str(e)); ax.axis("off")
            continue

        board = np.array(frame["board_after"])
        last = (frame["action_y"], frame["action_x"])
        lines = frame.get("ep_lines_cleared", "?")
        pieces = frame.get("ep_pieces_placed", "?")

        draw_board(ax, board, last_action=None,
                   title=f"{title}\nлиний: {lines}, фигур: {pieces}")

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_FILLED, edgecolor=C_GRID, label="Занятая клетка"),
        Patch(facecolor=C_EMPTY,  edgecolor=C_GRID, label="Пустая клетка"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"Состояние доски $16\\times16$ — эпизод {args.episode}, шаг {args.step}",
        fontsize=12, y=1.01,
    )

    out_name = f"fig_diagonal_step{args.step}.png"
    out = os.path.join(OUT_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  ✓ results/charts/report/{out_name}")


if __name__ == "__main__":
    main()
