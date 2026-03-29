"""
evaluate.py — сравнение агентов на одинаковых эпизодах.

Запуск:
    # Только бейзлайны:
    python evaluate.py

    # Бейзлайны + обученная PPO-модель:
    python evaluate.py --model ./models/block_puzzle_agent.zip

    # Настройка числа эпизодов:
    python evaluate.py --n 500 --seed 0 --model ./models/block_puzzle_agent.zip

Метрики:
    reward   — суммарная награда за эпизод
    lines    — очищенных линий за эпизод
    pieces   — размещённых фигур за эпизод
    perfect  — perfect clears за эпизод
    steps    — шагов (ходов) до конца игры
"""

from __future__ import annotations

import argparse
import time
import numpy as np
from tqdm import tqdm

from block_puzzle_env.environment import BlockPuzzleEnv
from baselines import RandomAgent, HeuristicAgent


# ================================================================== #
#  Запуск одного агента
# ================================================================== #

def run_episodes(agent, n: int = 200, seed: int = 42) -> dict[str, tuple[float, float]]:
    """
    Прогоняет агента на n эпизодах с фиксированными сидами (воспроизводимость).
    Все агенты используют одинаковые начальные состояния для честного сравнения.

    Returns:
        Словарь {метрика: (mean, std)}
    """
    env = BlockPuzzleEnv()

    rewards:  list[float] = []
    lines:    list[int]   = []
    pieces:   list[int]   = []
    perfects: list[int]   = []
    steps:    list[int]   = []

    for ep in tqdm(range(n), desc=f"  {agent.__class__.__name__:<18}", leave=False, ncols=70):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(env)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            done = terminated or truncated

        rewards.append(ep_reward)
        lines.append(info["ep_lines_cleared"])
        pieces.append(info["ep_pieces_placed"])
        perfects.append(info["ep_perfect_clears"])
        steps.append(env._step_count)

    env.close()

    return {
        "reward":  (float(np.mean(rewards)),  float(np.std(rewards))),
        "lines":   (float(np.mean(lines)),    float(np.std(lines))),
        "pieces":  (float(np.mean(pieces)),   float(np.std(pieces))),
        "perfect": (float(np.mean(perfects)), float(np.std(perfects))),
        "steps":   (float(np.mean(steps)),    float(np.std(steps))),
    }


# ================================================================== #
#  Вывод
# ================================================================== #

METRIC_LABELS = {
    "reward":  "Reward",
    "lines":   "Lines cleared",
    "pieces":  "Pieces placed",
    "perfect": "Perfect clears",
    "steps":   "Steps (moves)",
}

def print_agent_summary(name: str, stats: dict):
    print(f"\n  {'─' * 46}")
    print(f"  {name}")
    print(f"  {'─' * 46}")
    for key, label in METRIC_LABELS.items():
        m, s = stats[key]
        print(f"  {label:<18}  {m:>9.2f}  +/-  {s:.2f}")


def print_comparison_table(all_stats: dict[str, dict]):
    agents = list(all_stats.keys())
    col_w  = 22

    sep_len = 18 + col_w * len(agents)
    print(f"\n\n  {'=' * sep_len}")
    print(f"  COMPARATIVE TABLE  (mean +/- std)")
    print(f"  {'=' * sep_len}")

    header = f"  {'Metric':<18}" + "".join(f"{a:>{col_w}}" for a in agents)
    print(header)
    print(f"  {'-' * sep_len}")

    for key, label in METRIC_LABELS.items():
        best_mean = max(st[key][0] for st in all_stats.values())
        row = f"  {label:<18}"
        for stats in all_stats.values():
            m, s = stats[key]
            marker = "* " if m == best_mean else "  "
            cell = f"{marker}{m:>7.2f} +/-{s:>6.2f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"  {'=' * sep_len}")
    print("  * = best value for metric\n")


# ================================================================== #
#  Главная функция
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Block Puzzle RL — agent comparison"
    )
    parser.add_argument("--n",     type=int, default=200,
                        help="Number of episodes (default: 200)")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Base seed for reproducibility (default: 42)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO .zip model (optional)")
    args = parser.parse_args()

    # ---------------------------------------------------------------- #
    # Сборка агентов
    # ---------------------------------------------------------------- #
    agents: dict = {
        "Random":    RandomAgent(),
        "Heuristic": HeuristicAgent(),
    }

    if args.model:
        try:
            from sb3_contrib import MaskablePPO

            ppo_model = MaskablePPO.load(args.model)

            class PPOAgent:
                """Thin wrapper: exposes select_action() interface for MaskablePPO."""

                def select_action(self, env: BlockPuzzleEnv) -> int:
                    obs  = env._get_obs()[np.newaxis]        # (1, 4, 8, 8)
                    mask = env.action_masks()[np.newaxis]    # (1, 192)
                    action, _ = ppo_model.predict(
                        obs, action_masks=mask, deterministic=True
                    )
                    return int(action)

            agents["PPO (MaskablePPO)"] = PPOAgent()
            print(f"[evaluate] Model loaded: {args.model}")
        except Exception as e:
            print(f"[evaluate] Could not load model: {e}")

    # ---------------------------------------------------------------- #
    # Запуск оценки
    # ---------------------------------------------------------------- #
    print(f"\n{'=' * 60}")
    print(f"  AGENT EVALUATION")
    print(f"  Episodes: {args.n}  |  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    all_stats: dict[str, dict] = {}

    for name, agent in agents.items():
        t0 = time.time()
        stats = run_episodes(agent, n=args.n, seed=args.seed)
        elapsed = time.time() - t0
        print(f"  [{name}] done in {elapsed:.1f}s")
        print_agent_summary(name, stats)
        all_stats[name] = stats

    if len(all_stats) > 1:
        print_comparison_table(all_stats)


if __name__ == "__main__":
    main()
