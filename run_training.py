"""
run_training.py — единая точка запуска обучения для MLP и CNN архитектур.

Использование:
    # MLP (по умолчанию)
    python run_training.py --arch mlp

    # CNN с кастомным SmallCNN экстрактором
    python run_training.py --arch cnn

    # Задать имя запуска вручную (для TensorBoard и имени модели)
    python run_training.py --arch mlp --run-name mlp_baseline_v1

    # Переопределить число шагов и окружений
    python run_training.py --arch cnn --timesteps 2000000 --n-envs 4

    # Продолжить обучение с чекпоинта
    python run_training.py --arch mlp --resume ./models/checkpoints/block_puzzle_100000_steps.zip

Структура файлов после запуска:
    ./runs/<run_name>/              — TensorBoard логи + training_console.log
    ./models/checkpoints/           — чекпоинты каждые checkpoint_freq шагов
    ./models/<run_name>_final.zip   — финальная модель
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from config import REWARD, ENV, LOGGING, MLP_TRAIN, CNN_TRAIN
from block_puzzle_env.environment import BlockPuzzleEnv
from logger import TrainingLogger


# ================================================================== #
#  Фабрика окружений
# ================================================================== #

def _make_mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()


def _make_env(rank: int = 0, seed: int = 0):
    def _init():
        env = BlockPuzzleEnv()
        env = ActionMasker(env, _make_mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init


# ================================================================== #
#  TensorBoard callback
# ================================================================== #

class EpisodeStatsCallback(BaseCallback):
    """
    Логирует ep_lines_cleared, ep_perfect_clears, ep_pieces_placed
    ТОЛЬКО при завершении эпизода (dones=True).
    """

    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self._log_interval = log_interval
        self._lines:   list[int] = []
        self._perfect: list[int] = []
        self._placed:  list[int] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for info, done in zip(infos, dones):
            if done and "ep_lines_cleared" in info:
                self._lines.append(info["ep_lines_cleared"])
                self._perfect.append(info["ep_perfect_clears"])
                self._placed.append(info["ep_pieces_placed"])

        if len(self._lines) >= self._log_interval:
            self.logger.record("episode/lines_cleared_mean",  np.mean(self._lines))
            self.logger.record("episode/perfect_clears_mean", np.mean(self._perfect))
            self.logger.record("episode/pieces_placed_mean",  np.mean(self._placed))
            self._lines.clear()
            self._perfect.clear()
            self._placed.clear()

        return True


# ================================================================== #
#  Сборка policy_kwargs
# ================================================================== #

def _build_policy_kwargs(train_cfg: dict) -> dict:
    """
    Собирает policy_kwargs в зависимости от типа архитектуры.

    MLP:
        {"net_arch": [256, 256]}
        FlattenExtractor используется по умолчанию — явно указывать не нужно.

    CNN:
        {"net_arch": [256, 256],
         "features_extractor_class": SmallCNN,
         "features_extractor_kwargs": {"features_dim": 256}}
        SmallCNN заменяет NatureCNN, которая не работает на 8×8.
    """
    arch = train_cfg["features_extractor"]
    net_arch = train_cfg.get("net_arch", [256, 256])

    if arch == "flatten":
        return {"net_arch": net_arch}

    if arch == "cnn":
        from cnn_extractor import SmallCNN
        return {
            "net_arch": net_arch,
            "features_extractor_class":  SmallCNN,
            "features_extractor_kwargs": {
                "features_dim": train_cfg.get("features_dim", 256),
            },
        }

    raise ValueError(f"Неизвестный тип features_extractor: '{arch}'. Допустимо: 'flatten', 'cnn'.")


# ================================================================== #
#  Основная функция обучения
# ================================================================== #

def train(
    arch: str,
    run_name: str | None = None,
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    resume_path: str | None = None,
) -> None:
    """
    Args:
        arch:             "mlp" или "cnn"
        run_name:         имя запуска (TensorBoard + имя файла модели).
                          Если None — генерируется автоматически.
        total_timesteps:  переопределяет значение из конфига.
        n_envs:           переопределяет значение из конфига.
        resume_path:      путь к .zip чекпоинту для продолжения обучения.
    """
    # --- Выбор конфига ---
    if arch == "mlp":
        train_cfg = dict(MLP_TRAIN)
    elif arch == "cnn":
        train_cfg = dict(CNN_TRAIN)
    else:
        raise ValueError(f"Неизвестная архитектура: '{arch}'. Допустимо: 'mlp', 'cnn'.")

    # --- Переопределение параметров из CLI ---
    if total_timesteps is not None:
        train_cfg["total_timesteps"] = total_timesteps
    if n_envs is not None:
        train_cfg["n_envs"] = n_envs

    # --- Имя запуска ---
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{arch}_{timestamp}"

    save_path = os.path.join(LOGGING["save_dir"], f"{run_name}_final")

    # --- Создание директорий ---
    os.makedirs(LOGGING["checkpoint_dir"], exist_ok=True)
    os.makedirs(LOGGING["save_dir"], exist_ok=True)

    n_envs_actual = train_cfg["n_envs"]
    total_ts      = train_cfg["total_timesteps"]

    print(f"[run_training] arch={arch.upper()}  run_name={run_name}")
    print(f"[run_training] Создаём {n_envs_actual} параллельных окружений...")

    # --- Векторное окружение ---
    vec_env = SubprocVecEnv([_make_env(rank=i, seed=42) for i in range(n_envs_actual)])
    vec_env = VecMonitor(vec_env)

    # --- policy_kwargs ---
    policy_kwargs = _build_policy_kwargs(train_cfg)

    # --- Модель ---
    if resume_path:
        print(f"[run_training] Загружаем чекпоинт: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=vec_env,
            tensorboard_log=LOGGING["tensorboard_log"],
        )
        reset_num_timesteps = False
    else:
        model = MaskablePPO(
            policy=train_cfg["policy"],
            env=vec_env,
            learning_rate=train_cfg["learning_rate"],
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            n_epochs=train_cfg["n_epochs"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            ent_coef=train_cfg["ent_coef"],
            vf_coef=train_cfg["vf_coef"],
            max_grad_norm=train_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=LOGGING["tensorboard_log"],
            verbose=1,
        )
        reset_num_timesteps = True

    # --- Callbacks ---
    callbacks = [
        CheckpointCallback(
            save_freq=max(LOGGING["checkpoint_freq"] // n_envs_actual, 1),
            save_path=LOGGING["checkpoint_dir"],
            name_prefix=f"{arch}_puzzle",
            verbose=1,
        ),
        EpisodeStatsCallback(log_interval=LOGGING["log_interval"]),
    ]

    # --- Лог-файл ---
    log_file_path = os.path.join(
        LOGGING["tensorboard_log"], run_name, "training_console.log"
    )

    with TrainingLogger(log_file_path) as t_logger:
        t_logger.log_header()
        t_logger.log_model_info(model)
        t_logger.log_params({
            "ARCH":   arch.upper(),
            "TRAIN":  train_cfg,
            "REWARD": REWARD,
            "ENV":    ENV,
        })
        t_logger.start_capture()

        try:
            print(f"[run_training] Начинаем обучение на {total_ts:,} шагов...")

            model.learn(
                total_timesteps=total_ts,
                callback=callbacks,
                tb_log_name=run_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=True,
            )
            model.save(save_path)
            print(f"[run_training] Модель сохранена: {save_path}.zip")

        finally:
            t_logger.stop_capture()

    vec_env.close()


# ================================================================== #
#  CLI
# ================================================================== #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block Puzzle RL — запуск обучения MLP или CNN агента",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["mlp", "cnn"],
        default="mlp",
        help=(
            "Архитектура агента:\n"
            "  mlp  — MlpPolicy + FlattenExtractor (вход: flatten 4×8×8=256)\n"
            "  cnn  — MlpPolicy + SmallCNN (3 свёрточных слоя, вход: 4×8×8)\n"
            "(default: mlp)"
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Имя запуска для TensorBoard и имени файла модели.\n"
            "Если не задано, генерируется автоматически: <arch>_<timestamp>."
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        metavar="N",
        help="Переопределить total_timesteps из конфига.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        metavar="N",
        help="Переопределить число параллельных окружений из конфига.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Путь к .zip чекпоинту для продолжения обучения.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        arch=args.arch,
        run_name=args.run_name,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        resume_path=args.resume,
    )
