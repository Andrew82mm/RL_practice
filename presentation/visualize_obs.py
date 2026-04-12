"""
visualize_obs.py — визуализация тензора наблюдений (14, 8, 8).

Запуск:
    # Случайная игра, первый кадр:
    python visualize_obs.py

    # Конкретный seed и шаг:
    python visualize_obs.py --seed 7 --steps 12

    # Сохранить в файл:
    python visualize_obs.py --seed 7 --steps 12 --save obs.png

    # Показать только конкретные каналы:
    python visualize_obs.py --channels 0 6 7 8 9 12 13
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from evaluation.baselines import RandomAgent
from block_puzzle_env.environment import BlockPuzzleEnv

# ------------------------------------------------------------------
# Метаданные каналов
# ------------------------------------------------------------------

CHANNEL_META = [
    # (название,        короткое описание,              colormap)
    ("Board",           "Занятые клетки",               "Greys"),
    ("Piece 1 shape",   "Форма фигуры 1",               "Blues"),
    ("Piece 2 shape",   "Форма фигуры 2",               "Oranges"),
    ("Piece 3 shape",   "Форма фигуры 3",               "Greens"),
    ("Row fill",        "Заполненность строк",          "YlOrRd"),
    ("Col fill",        "Заполненность столбцов",       "YlOrRd"),
    ("Heatmap 1",       "Куда можно поставить фигуру 1","Blues"),
    ("Heatmap 2",       "Куда можно поставить фигуру 2","Oranges"),
    ("Heatmap 3",       "Куда можно поставить фигуру 3","Greens"),
    ("Survivability 1", "Lookahead: фигура 1",          "RdYlGn"),
    ("Survivability 2", "Lookahead: фигура 2",          "RdYlGn"),
    ("Survivability 3", "Lookahead: фигура 3",          "RdYlGn"),
    ("Blob map",        "Крупнейшая пустая область",    "cool"),
    ("Dead zones",      "Мёртвые зоны",                 "Reds"),
]


def get_game_state(seed: int, n_steps: int) -> tuple[np.ndarray, BlockPuzzleEnv]:
    """Запускает случайного агента n_steps шагов и возвращает obs + env."""
    env = BlockPuzzleEnv()
    agent = RandomAgent()
    obs, _ = env.reset(seed=seed)

    for _ in range(n_steps):
        action = agent.select_action(env)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 1)

    return env._get_obs(), env


def draw_grid_lines(ax, n: int = 8):
    """Рисует тонкую сетку поверх heatmap."""
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.4, alpha=0.5)
        ax.axvline(i - 0.5, color="white", linewidth=0.4, alpha=0.5)


def visualize(
    obs: np.ndarray,
    env: BlockPuzzleEnv,
    channels: list[int] | None = None,
    save_path: str | None = None,
):
    if channels is None:
        channels = list(range(14))

    ncols = 7
    nrows = (len(channels) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.2, nrows * 2.4),
        facecolor="#1a1a2e",
    )
    axes = np.array(axes).reshape(-1)

    for ax_idx, ch in enumerate(channels):
        ax = axes[ax_idx]
        data = obs[ch]
        name, desc, cmap_name = CHANNEL_META[ch]

        # Для бинарных каналов (board, pieces, blob, dead) используем резкий cmap
        vmin, vmax = 0.0, max(data.max(), 1e-6)

        im = ax.imshow(
            data,
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        draw_grid_lines(ax)

        # Заголовок
        ax.set_title(
            f"Ch {ch}: {name}\n{desc}",
            fontsize=7,
            color="white",
            pad=3,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Цветовая шкала справа
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5, colors="white")
        cbar.outline.set_edgecolor("white")

    # Скрываем лишние оси
    for ax in axes[len(channels):]:
        ax.set_visible(False)

    # Информация об игровом состоянии
    pieces_str = ", ".join(
        str(env.current_pieces[i]) for i in range(len(env.current_pieces))
    )
    lines = env._ep_lines_cleared
    placed = env._ep_pieces_placed

    fig.suptitle(
        f"Observation tensor  |  shape: {obs.shape}  |  "
        f"lines: {lines}  pieces: {placed}  current_pieces: [{pieces_str}]",
        fontsize=10,
        color="white",
        y=1.01,
    )
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"[visualize_obs] Сохранено: {save_path}")
    else:
        import matplotlib
        if matplotlib.get_backend() == "agg" or not _has_display():
            default_path = "obs.png"
            plt.savefig(default_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
            print(f"[visualize_obs] GUI недоступен. Сохранено: {default_path}")
        else:
            plt.show()

    plt.close(fig)


def _has_display() -> bool:
    """Проверяет наличие GUI-дисплея."""
    import os
    import sys
    if sys.platform == "linux" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Визуализация тензора наблюдений (14,8,8)")
    parser.add_argument("--seed",     type=int, default=42,  help="Random seed")
    parser.add_argument("--steps",    type=int, default=15,  help="Шагов до снимка состояния")
    parser.add_argument("--save",     type=str, default=None,help="Путь для сохранения PNG")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="Какие каналы показать (default: все 0-13)")
    args = parser.parse_args()

    print(f"[visualize_obs] seed={args.seed}, steps={args.steps}")
    obs, env = get_game_state(args.seed, args.steps)
    print(f"[visualize_obs] obs shape: {obs.shape}, board filled: {obs[0].sum():.0f}/64")
    print(f"[visualize_obs] survivability max: {obs[9:12].max():.3f}")
    print(f"[visualize_obs] blob cells: {obs[12].sum():.0f}, dead cells: {obs[13].sum():.0f}")

    visualize(obs, env, channels=args.channels, save_path=args.save)


if __name__ == "__main__":
    main()
