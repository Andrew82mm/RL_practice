"""
sweep.py — эмпирический подбор гиперпараметров перед полноценным обучением.

Запускает N конфигураций по 100k шагов (configurable), после каждого прогона
дополнительно оценивает модель на M эпизодах через evaluate.py логику.
В конце печатает ранжированную сравнительную таблицу по выбранной метрике.

Использование:
    # Запустить все заданные конфигурации (100k шагов, 8 envs по умолчанию):
    python sweep.py

    # Уменьшить timesteps и eval-эпизоды для быстрого прохода:
    python sweep.py --timesteps 50000 --eval-n 50

    # Запустить только конкретные конфиги по индексу (0-based):
    python sweep.py --configs 0 2 4

    # Изменить метрику ранжирования:
    python sweep.py --rank-by lines

    # Продолжить прерванный sweep (пропустить уже готовые модели):
    python sweep.py --skip-existing

Структура вывода после запуска:
    ./models/sweep/<run_name>_final.zip   — модели всех конфигураций
    ./runs/<run_name>/                    — TensorBoard логи
    sweep_results.txt                     — итоговая таблица (дублируется в файл)
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

# ── Добавим путь к проекту в sys.path, если запускаем из другой директории
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_puzzle_env.environment import BlockPuzzleEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import gymnasium as gym


# ================================================================== #
#  КОНФИГУРАЦИИ ДЛЯ SWEEP
#  Редактируй этот список — каждый dict описывает один эксперимент.
# ================================================================== #

SWEEP_CONFIGS: list[dict[str, Any]] = [

    # ── 0. Baseline ──────────────────
    {
        "name": "baseline",
        "reward": {
            "place_piece":       0.05,
            "line_cleared":      3.0,
            "combo_multiplier":  2.0,
            "perfect_clear":    15.0,
            "invalid_move":     -2.0,
            "game_over":        -3.0,
            "game_over_per_cell": 0.0,
        },
        "train": {
            "ent_coef":      0.03,
            "learning_rate": 3e-4,
            "net_arch":     [256, 256],
        },
    },

    # ── 1. Усиленный штраф за game over ─────────────────────────────
    # Гипотеза: сильнее штрафовать за конец игры → агент осторожнее
    {
        "name": "heavy_gameover",
        "reward": {
            "place_piece":       0.05,
            "line_cleared":      3.0,
            "combo_multiplier":  2.0,
            "perfect_clear":    15.0,
            "invalid_move":     -2.0,
            "game_over":       -10.0,   # ← усилен
            "game_over_per_cell": 0.0,
        },
        "train": {
            "ent_coef":      0.03,
            "learning_rate": 3e-4,
            "net_arch":     [256, 256],
        },
    },

    # ── 2. Штраф за занятые клетки при game over ─────────────────────
    # Гипотеза: штраф за каждую занятую клетку → очищать больше
    {
        "name": "per_cell_penalty",
        "reward": {
            "place_piece":       0.05,
            "line_cleared":      3.0,
            "combo_multiplier":  2.0,
            "perfect_clear":    15.0,
            "invalid_move":     -2.0,
            "game_over":        -3.0,
            "game_over_per_cell": -0.2,  # ← включён
        },
        "train": {
            "ent_coef":      0.03,
            "learning_rate": 3e-4,
            "net_arch":     [256, 256],
        },
    },

    # ── 3. Усиленная награда за линии + слабый combo ──────────────────
    # Гипотеза: линейная награда без combo → проще для обучения на старте
    {
        "name": "linear_lines",
        "reward": {
            "place_piece":       0.05,
            "line_cleared":      5.0,    # ← выше
            "combo_multiplier":  1.0,    # ← combo отключён (×1)
            "perfect_clear":    15.0,
            "invalid_move":     -2.0,
            "game_over":        -3.0,
            "game_over_per_cell": 0.0,
        },
        "train": {
            "ent_coef":      0.03,
            "learning_rate": 3e-4,
            "net_arch":     [256, 256],
        },
    },

    # ── 4. Высокая энтропия → больше exploration ─────────────────────
    # Гипотеза: ent_coef=0.1 помогает выбраться из локальных минимумов
    {
        "name": "high_entropy",
        "reward": {
            "place_piece":       0.05,
            "line_cleared":      3.0,
            "combo_multiplier":  2.0,
            "perfect_clear":    15.0,
            "invalid_move":     -2.0,
            "game_over":        -3.0,
            "game_over_per_cell": 0.0,
        },
        "train": {
            "ent_coef":      0.10,     # ← выше
            "learning_rate": 3e-4,
            "net_arch":     [256, 256],
        },
    },

    # ── 5. Нет награды за размещение, только за линии ────────────────
    # Гипотеза: убрать шум от place_piece → чище gradient signal
    {
        "name": "no_place_reward",
        "reward": {
            "place_piece":       0.0,    # ← убран
            "line_cleared":      3.0,
            "combo_multiplier":  2.0,
            "perfect_clear":    15.0,
            "invalid_move":     -2.0,
            "game_over":        -3.0,
            "game_over_per_cell": 0.0,
        },
        "train": {
            "ent_coef":      0.03,
            "learning_rate": 3e-4,
            "net_arch":     [256, 256],
        },
    },

    # ── 6. Более глубокая сеть ────────────────────────────────────────
    # Гипотеза: 3 слоя лучше справятся с планированием
    {
        "name": "deeper_mlp",
        "reward": {
            "place_piece":       0.05,
            "line_cleared":      3.0,
            "combo_multiplier":  2.0,
            "perfect_clear":    15.0,
            "invalid_move":     -2.0,
            "game_over":        -3.0,
            "game_over_per_cell": 0.0,
        },
        "train": {
            "ent_coef":      0.03,
            "learning_rate": 3e-4,
            "net_arch":     [512, 512, 256],  # ← глубже и шире
        },
    },
]


# ================================================================== #
#  Общие PPO-параметры (не варьируются в sweep)
# ================================================================== #

PPO_BASE = {
    "n_envs":        4,     # меньше чем при финальном обучении — экономия времени
    "n_steps":    2048,
    "batch_size":  512,
    "n_epochs":     10,
    "gamma":      0.99,
    "gae_lambda": 0.95,
    "clip_range":  0.2,
    "vf_coef":     0.5,
    "max_grad_norm": 0.5,
}


# ================================================================== #
#  Вспомогательный callback: собирает метрики для sweep-таблицы
# ================================================================== #

class SweepMetricsCallback(BaseCallback):
    """
    Собирает ep_lines_cleared, ep_pieces_placed при done=True.
    Хранит все эпизодные значения для вычисления статистики после обучения.
    """
    def __init__(self):
        super().__init__()
        self.ep_lines:   list[int] = []
        self.ep_pieces:  list[int] = []
        self.ep_perfect: list[int] = []

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [])
        ):
            if done and "ep_lines_cleared" in info:
                self.ep_lines.append(info["ep_lines_cleared"])
                self.ep_pieces.append(info["ep_pieces_placed"])
                self.ep_perfect.append(info["ep_perfect_clears"])
        return True

    def summary(self) -> dict[str, float]:
        """Возвращает средние за последние 200 эпизодов (финальная производительность)."""
        n = min(200, len(self.ep_lines))
        if n == 0:
            return {"lines": 0.0, "pieces": 0.0, "perfect": 0.0, "episodes": 0}
        return {
            "lines":   float(np.mean(self.ep_lines[-n:])),
            "pieces":  float(np.mean(self.ep_pieces[-n:])),
            "perfect": float(np.mean(self.ep_perfect[-n:])),
            "episodes": len(self.ep_lines),
        }


# ================================================================== #
#  Фабрика окружений с инжектированным REWARD
# ================================================================== #

def make_env_factory(reward_cfg: dict, rank: int = 0, seed: int = 42):
    """
    Создаёт среду с подменённой конфигурацией наград.
    BlockPuzzleEnv читает REWARD из модуля config — временно патчим его.
    """
    def _init():
        # Патчим глобальный REWARD перед созданием среды
        import config as cfg
        original_reward = dict(cfg.REWARD)
        cfg.REWARD.update(reward_cfg)

        env = BlockPuzzleEnv()

        # Восстанавливаем оригинал после создания
        cfg.REWARD.clear()
        cfg.REWARD.update(original_reward)

        env = ActionMasker(env, lambda e: e.action_masks())
        env.reset(seed=seed + rank)
        return env
    return _init


# ================================================================== #
#  Оценка обученной модели
# ================================================================== #

def quick_eval(model_path: str, n_episodes: int, seed: int) -> dict[str, float]:
    """
    Запускает обученную модель на n_episodes и возвращает средние метрики.
    Использует одиночную (не-векторную) среду для детерминированной оценки.
    """
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        return {"lines": -1.0, "pieces": -1.0, "perfect": -1.0, "error": str(e)}

    env = BlockPuzzleEnv()
    lines_list, pieces_list, perfect_list = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            mask = env.action_masks()[np.newaxis]
            action, _ = model.predict(obs[np.newaxis], action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        lines_list.append(info["ep_lines_cleared"])
        pieces_list.append(info["ep_pieces_placed"])
        perfect_list.append(info["ep_perfect_clears"])

    env.close()
    return {
        "lines":   float(np.mean(lines_list)),
        "pieces":  float(np.mean(pieces_list)),
        "perfect": float(np.mean(perfect_list)),
    }


# ================================================================== #
#  Запуск одной конфигурации
# ================================================================== #

def run_config(
    cfg: dict[str, Any],
    timesteps: int,
    eval_n: int,
    eval_seed: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """
    Обучает модель с заданной конфигурацией и оценивает её.
    Возвращает словарь с результатами для итоговой таблицы.
    """
    run_name = f"sweep_{cfg['name']}"
    save_path = os.path.join("./models/sweep", f"{run_name}_final")
    model_file = save_path + ".zip"

    result: dict[str, Any] = {
        "name":      cfg["name"],
        "run_name":  run_name,
        "reward":    cfg["reward"],
        "train":     cfg["train"],
        "skipped":   False,
        "error":     None,
        "train_metrics": {},
        "eval_metrics":  {},
        "elapsed_s": 0.0,
    }

    # ── Пропустить если модель уже существует ────────────────────────
    if skip_existing and os.path.exists(model_file):
        print(f"  [skip] {run_name} — модель уже существует")
        result["skipped"] = True
        result["eval_metrics"] = quick_eval(model_file, eval_n, eval_seed)
        return result

    os.makedirs("./models/sweep", exist_ok=True)

    reward_cfg = cfg["reward"]
    train_overrides = cfg["train"]

    n_envs = PPO_BASE.get("n_envs", 4)
    t0 = time.time()

    try:
        # ── Среды ────────────────────────────────────────────────────
        vec_env = SubprocVecEnv([
            make_env_factory(reward_cfg, rank=i, seed=42)
            for i in range(n_envs)
        ])
        vec_env = VecMonitor(vec_env)

        # ── policy_kwargs ─────────────────────────────────────────────
        net_arch = train_overrides.get("net_arch", [256, 256])
        policy_kwargs = {"net_arch": net_arch}

        # ── Модель ───────────────────────────────────────────────────
        model = MaskablePPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=train_overrides.get("learning_rate", PPO_BASE.get("learning_rate", 3e-4)),
            n_steps=PPO_BASE["n_steps"],
            batch_size=PPO_BASE["batch_size"],
            n_epochs=PPO_BASE["n_epochs"],
            gamma=PPO_BASE["gamma"],
            gae_lambda=PPO_BASE["gae_lambda"],
            clip_range=PPO_BASE["clip_range"],
            ent_coef=train_overrides.get("ent_coef", 0.03),
            vf_coef=PPO_BASE["vf_coef"],
            max_grad_norm=PPO_BASE["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log="./runs/",
            verbose=0,   # тихий режим — не засоряем консоль sweep
        )

        # ── Callback ─────────────────────────────────────────────────
        metrics_cb = SweepMetricsCallback()

        # ── Обучение ─────────────────────────────────────────────────
        model.learn(
            total_timesteps=timesteps,
            callback=metrics_cb,
            tb_log_name=run_name,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        model.save(save_path)
        vec_env.close()

        result["train_metrics"] = metrics_cb.summary()

    except Exception as e:
        result["error"] = str(e)
        print(f"  [ERROR] {run_name}: {e}")
        return result

    result["elapsed_s"] = time.time() - t0

    # ── Независимая оценка обученной модели ──────────────────────────
    print(f"  [eval] Оцениваем {run_name} на {eval_n} эпизодах...")
    result["eval_metrics"] = quick_eval(model_file, eval_n, eval_seed)

    return result


# ================================================================== #
#  Форматирование таблицы результатов
# ================================================================== #

def format_results(results: list[dict], rank_by: str) -> str:
    """Форматирует результаты в читаемую таблицу и ранжирует по метрике."""
    # Фильтруем упавшие
    ok = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    # Сортировка по eval-метрике (если есть), иначе по train-метрике
    def sort_key(r):
        em = r.get("eval_metrics", {})
        tm = r.get("train_metrics", {})
        return em.get(rank_by, tm.get(rank_by, -1))

    ok.sort(key=sort_key, reverse=True)

    lines = []
    sep = "=" * 98

    lines.append(sep)
    lines.append(f"  SWEEP RESULTS  |  Sorted by eval/{rank_by}  |  {len(results)} configs")
    lines.append(sep)

    # Шапка
    lines.append(
        f"  {'#':<3} {'Name':<22} {'Train lines':>12} {'Train pieces':>13} "
        f"{'Eval lines':>11} {'Eval pieces':>12} {'Time':>8}"
    )
    lines.append("-" * 98)

    for i, r in enumerate(ok, 1):
        tm = r.get("train_metrics", {})
        em = r.get("eval_metrics", {})
        skipped_mark = " [cached]" if r.get("skipped") else ""
        lines.append(
            f"  {i:<3} {r['name']:<22}"
            f"  {tm.get('lines', 0):>10.2f}"
            f"  {tm.get('pieces', 0):>11.2f}"
            f"  {em.get('lines', 0):>10.2f}"
            f"  {em.get('pieces', 0):>11.2f}"
            f"  {r['elapsed_s']:>6.0f}s"
            f"{skipped_mark}"
        )

    lines.append("-" * 98)

    # Победитель
    if ok:
        winner = ok[0]
        lines.append(f"\n  WINNER: {winner['name']}")
        lines.append(f"  Eval lines: {winner['eval_metrics'].get('lines', 0):.2f}  |  "
                     f"Eval pieces: {winner['eval_metrics'].get('pieces', 0):.2f}  |  "
                     f"Eval perfect: {winner['eval_metrics'].get('perfect', 0):.3f}")

        lines.append(f"\n  Winning reward config:")
        for k, v in winner["reward"].items():
            lines.append(f"    {k}: {v}")
        lines.append(f"\n  Winning train overrides:")
        for k, v in winner["train"].items():
            lines.append(f"    {k}: {v}")

    # Упавшие
    if failed:
        lines.append(f"\n  FAILED ({len(failed)}):")
        for r in failed:
            lines.append(f"    {r['name']}: {r['error']}")

    lines.append(sep)
    return "\n".join(lines)


# ================================================================== #
#  Главная функция
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Sweep гиперпараметров для Block Puzzle RL",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Шагов обучения на конфигурацию (default: 100000)"
    )
    parser.add_argument(
        "--eval-n", type=int, default=100,
        help="Эпизодов для оценки каждой модели после обучения (default: 100)"
    )
    parser.add_argument(
        "--eval-seed", type=int, default=42,
        help="Seed для eval-эпизодов (default: 42)"
    )
    parser.add_argument(
        "--configs", type=int, nargs="+", default=None,
        metavar="IDX",
        help=(
            "Индексы конфигов для запуска (0-based).\n"
            "По умолчанию — все. Пример: --configs 0 2 4"
        )
    )
    parser.add_argument(
        "--rank-by", type=str, default="lines",
        choices=["lines", "pieces", "perfect"],
        help="Метрика для ранжирования в итоговой таблице (default: lines)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Пропускать конфиги, для которых модель уже сохранена"
    )
    parser.add_argument(
        "--output", type=str, default="sweep_results.txt",
        help="Файл для сохранения итоговой таблицы (default: sweep_results.txt)"
    )
    args = parser.parse_args()

    # ── Выбор конфигов ───────────────────────────────────────────────
    configs_to_run = SWEEP_CONFIGS
    if args.configs is not None:
        configs_to_run = [SWEEP_CONFIGS[i] for i in args.configs]

    print(f"\n{'='*60}")
    print(f"  HYPERPARAMETER SWEEP")
    print(f"  Конфигов:  {len(configs_to_run)}")
    print(f"  Timesteps: {args.timesteps:,} на конфиг")
    print(f"  Eval:      {args.eval_n} эпизодов (seed={args.eval_seed})")
    print(f"  n_envs:    {PPO_BASE['n_envs']} (sweep режим)")
    print(f"{'='*60}\n")

    # ── Описание всех конфигов ───────────────────────────────────────
    for i, cfg in enumerate(configs_to_run):
        print(f"  [{i}] {cfg['name']}")
        delta_r = {k: v for k, v in cfg["reward"].items()
                   if v != SWEEP_CONFIGS[0]["reward"].get(k)}
        delta_t = {k: v for k, v in cfg["train"].items()
                   if v != SWEEP_CONFIGS[0]["train"].get(k)}
        if delta_r:
            print(f"       reward changes: {delta_r}")
        if delta_t:
            print(f"       train changes:  {delta_t}")
    print()

    # ── Запуск ───────────────────────────────────────────────────────
    all_results: list[dict] = []
    total_t0 = time.time()

    for i, cfg in enumerate(configs_to_run):
        print(f"[{i+1}/{len(configs_to_run)}] Запускаем: {cfg['name']}  "
              f"({args.timesteps:,} steps, {PPO_BASE['n_envs']} envs)")

        result = run_config(
            cfg=cfg,
            timesteps=args.timesteps,
            eval_n=args.eval_n,
            eval_seed=args.eval_seed,
            skip_existing=args.skip_existing,
        )
        all_results.append(result)

        # Промежуточный результат
        if result["error"] is None:
            tm = result["train_metrics"]
            em = result["eval_metrics"]
            print(
                f"  done in {result['elapsed_s']:.0f}s  |  "
                f"train: lines={tm.get('lines',0):.1f} pieces={tm.get('pieces',0):.1f}  |  "
                f"eval:  lines={em.get('lines',0):.1f} pieces={em.get('pieces',0):.1f}"
            )
        print()

    total_elapsed = time.time() - total_t0

    # ── Итоговая таблица ─────────────────────────────────────────────
    table = format_results(all_results, rank_by=args.rank_by)
    print(table)
    print(f"\n  Суммарное время: {total_elapsed/60:.1f} мин\n")

    # ── Сохранить таблицу в файл ─────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"Sweep: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Timesteps per config: {args.timesteps:,}\n")
        f.write(f"Total time: {total_elapsed/60:.1f} min\n\n")
        f.write(table)
    print(f"  Результаты сохранены в: {args.output}")
    print(f"  TensorBoard: tensorboard --logdir ./runs  (фильтр: sweep_*)\n")


if __name__ == "__main__":
    main()
