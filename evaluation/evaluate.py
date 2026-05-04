"""
evaluate.py — сравнение агентов на одинаковых эпизодах.

ВАЖНО: Если модель обучалась с VecNormalize (norm_reward=True),
при оценке нормализация наград НЕ нужна (оцениваем в оригинальных наградах),
но нормализация наблюдений (если была включена norm_obs=True) — нужна.
В run_training.py norm_obs=False, поэтому evaluate работает напрямую.

Запуск:
    # Только бейзлайны:
    python evaluate.py --n 500 --seed 42

    # PPO модель (без VecNormalize — norm_obs=False):
    python evaluate.py --model ./models/mlp_v2_final.zip --n 500 --seed 42

    # Все чекпоинты MLP (кривая обучения):
    python evaluate.py --checkpoints ./models/checkpoints/ --prefix mlp_puzzle --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from block_puzzle_env.environment import BlockPuzzleEnv
from evaluation.baselines import RandomAgent, HeuristicAgent


# ================================================================== #
#  Запуск одного агента
# ================================================================== #

def run_episodes(agent, n: int = 200, seed: int = 42) -> dict[str, tuple[float, float]]:
    """
    Прогоняет агента на n эпизодах с фиксированными сидами.
    Оценка всегда идёт в оригинальных (ненормированных) наградах.
    """
    env = BlockPuzzleEnv()
    rewards, lines, pieces, perfects, steps = [], [], [], [], []

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
#  PPO агент-обёртка
# ================================================================== #

class PPOAgent:
    """
    Обёртка над MaskablePPO для интерфейса select_action(env).
    Оценка идёт в оригинальном пространстве наград (norm_reward не применяется).
    norm_obs=False в run_training.py — поэтому obs передаём напрямую.
    """

    def __init__(self, model_path: str, name: str | None = None):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)
        self._name = name or os.path.basename(model_path)

    @property
    def __class__(self):
        # Для красивого вывода в tqdm
        class _Named:
            __name__ = self._name
        return _Named

    def select_action(self, env: BlockPuzzleEnv) -> int:
        import torch
        obs = env._get_obs()[np.newaxis]
        # Обрезаем obs до ожидаемого числа каналов (для совместимости со старыми моделями)
        expected_ch = self.model.observation_space.shape[0]
        if obs.shape[1] != expected_ch:
            obs = obs[:, :expected_ch, :, :]
        mask = env.action_masks()  # (n_actions,) bool

        # Считаем логиты напрямую, минуя SB3 distribution (обход Simplex validation в PyTorch 2.x)
        policy = self.model.policy
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            features = policy.extract_features(obs_t, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)[0].cpu().numpy()  # (n_actions,)

        logits[~mask] = -1e8  # маскируем невалидные действия
        return int(np.argmax(logits))


# ================================================================== #
#  Кривая обучения по чекпоинтам
# ================================================================== #

def eval_checkpoints(
    checkpoints_dir: str,
    prefix: str,
    n: int,
    seed: int,
) -> list[dict]:
    """
    Находит все чекпоинты с заданным префиксом, сортирует по числу шагов,
    прогоняет оценку и возвращает список результатов для построения кривой.
    """
    pattern = os.path.join(checkpoints_dir, f"{prefix}_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        print(f"[evaluate] Чекпоинты не найдены: {pattern}")
        return []

    # Извлекаем число шагов из имени файла и сортируем
    def _steps(path):
        m = re.search(r"_(\d+)_steps\.zip$", path)
        return int(m.group(1)) if m else 0

    files = sorted(files, key=_steps)
    print(f"[evaluate] Найдено чекпоинтов: {len(files)}")

    results = []
    for path in files:
        n_steps = _steps(path)
        print(f"  Оцениваем: {os.path.basename(path)} ({n_steps:,} steps)...")
        try:
            agent = PPOAgent(path, name=f"{prefix}@{n_steps//1000}k")
            stats = run_episodes(agent, n=n, seed=seed)
            results.append({
                "path":    path,
                "steps":   n_steps,
                "stats":   stats,
            })
            m_l = stats["lines"][0]
            m_p = stats["pieces"][0]
            print(f"    lines={m_l:.2f}  pieces={m_p:.2f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    return results


def print_learning_curve(results: list[dict], prefix: str):
    """Печатает кривую обучения в виде таблицы."""
    print(f"\n  {'='*60}")
    print(f"  КРИВАЯ ОБУЧЕНИЯ: {prefix}")
    print(f"  {'='*60}")
    print(f"  {'Steps':>10}  {'Lines':>8}  {'Pieces':>9}  {'Reward':>9}")
    print(f"  {'-'*46}")
    for r in results:
        s = r["steps"]
        l = r["stats"]["lines"][0]
        p = r["stats"]["pieces"][0]
        rw = r["stats"]["reward"][0]
        print(f"  {s:>10,}  {l:>8.2f}  {p:>9.2f}  {rw:>9.2f}")
    print(f"  {'='*60}\n")


# ================================================================== #
#  Печать результатов
# ================================================================== #

METRIC_LABELS = {
    "reward":  "Reward",
    "lines":   "Lines cleared",
    "pieces":  "Pieces placed",
    "perfect": "Perfect clears",
    "steps":   "Steps (moves)",
}


def print_agent_summary(name: str, stats: dict):
    print(f"\n  {'─' * 48}")
    print(f"  {name}")
    print(f"  {'─' * 48}")
    for key, label in METRIC_LABELS.items():
        m, s = stats[key]
        print(f"  {label:<18}  {m:>9.2f}  +/-  {s:.2f}")


def print_comparison_table(all_stats: dict[str, dict]):
    agents = list(all_stats.keys())
    col_w  = 22
    sep_len = 18 + col_w * len(agents)
    print(f"\n\n  {'='*sep_len}")
    print(f"  COMPARATIVE TABLE  (mean +/- std)")
    print(f"  {'='*sep_len}")
    header = f"  {'Metric':<18}" + "".join(f"{a:>{col_w}}" for a in agents)
    print(header)
    print(f"  {'-'*sep_len}")
    for key, label in METRIC_LABELS.items():
        best_mean = max(st[key][0] for st in all_stats.values())
        row = f"  {label:<18}"
        for stats in all_stats.values():
            m, s = stats[key]
            marker = "* " if m == best_mean else "  "
            cell = f"{marker}{m:>7.2f} +/-{s:>6.2f}"
            row += f"{cell:>{col_w}}"
        print(row)
    print(f"  {'='*sep_len}")
    print("  * = best value for metric\n")


# ================================================================== #
#  Главная функция
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Block Puzzle RL — оценка и сравнение агентов"
    )
    parser.add_argument("--n",     type=int, default=200, help="Число эпизодов (default: 200)")
    parser.add_argument("--seed",  type=int, default=42,  help="Базовый seed (default: 42)")
    parser.add_argument(
        "--model", type=str, default=None, metavar="PATH",
        help="Путь к .zip модели MaskablePPO (можно указать несколько через пробел с --models)"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None, metavar="PATH",
        help="Несколько моделей для сравнения: --models mlp.zip cnn.zip"
    )
    parser.add_argument(
        "--checkpoints", type=str, default=None, metavar="DIR",
        help="Директория с чекпоинтами для построения кривой обучения"
    )
    parser.add_argument(
        "--prefix", type=str, default="mlp_puzzle", metavar="PREFIX",
        help="Префикс имён чекпоинтов (default: mlp_puzzle)"
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Не запускать Random и Heuristic агентов"
    )
    args = parser.parse_args()

    # ── Режим: кривая обучения по чекпоинтам ────────────────────────
    if args.checkpoints:
        results = eval_checkpoints(
            args.checkpoints, args.prefix, n=args.n, seed=args.seed
        )
        print_learning_curve(results, prefix=args.prefix)
        return

    # ── Режим: сравнение агентов ─────────────────────────────────────
    agents: dict = {}

    if not args.no_baselines:
        agents["Random"]    = RandomAgent()
        agents["Heuristic"] = HeuristicAgent()

    # Одна или несколько моделей
    model_paths = []
    if args.model:
        model_paths.append(args.model)
    if args.models:
        model_paths.extend(args.models)

    for path in model_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            agents[name] = PPOAgent(path, name=name)
        except Exception as e:
            print(f"[evaluate] Не удалось загрузить {path}: {e}")

    print(f"\n{'='*60}")
    print(f"  AGENT EVALUATION")
    print(f"  Episodes: {args.n}  |  Seed: {args.seed}")
    print(f"{'='*60}\n")

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
