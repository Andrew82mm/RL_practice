"""
transfer_train.py — Transfer learning: Gen 2 CNN (9 каналов) → Gen 3 CNN (14 каналов).

Идея:
    Gen 2 CNN умеет играть на уровне ~8.7 lines.
    Gen 3 добавляет 5 новых каналов (9-13: survivability, blob map, dead zones).
    Вместо обучения с нуля — переносим веса Gen 2 и дообучаем.

Схема переноса весов:
    Conv1 weight: (32, 9, 3, 3) → (32, 14, 3, 3)
        каналы 0-8 = веса Gen 2 (полностью рабочие)
        каналы 9-13 = нули (модель сразу ведёт себя как Gen 2,
                             постепенно учится читать новые каналы)
    Все остальные слои (Conv2, Conv3, Linear, MLP heads) — копируем как есть,
    формы совпадают, перенос 1-в-1.

Запуск:
    python transfer_train.py
    python transfer_train.py --run-name cnn_gen3_v2 --timesteps 10000000
    python transfer_train.py --gen2-model ./models/gen2/cnn_gen_2/cnn_2_final.zip
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from block_puzzle_env.environment import BlockPuzzleEnv
from cnn_extractor import SmallCNN
from config import CNN_TRAIN, LOGGING, REWARD, ENV
from logger import TrainingLogger


GEN2_MODEL_DEFAULT = "./models/gen2/cnn_gen_2/cnn_2_final.zip"


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
#  Перенос весов
# ================================================================== #

def transfer_weights(gen2_model: MaskablePPO, gen3_model: MaskablePPO) -> None:
    """
    Переносит веса из Gen 2 (9 каналов) в Gen 3 (14 каналов).

    Conv1 weight shape:
        Gen 2: (out_ch=32, in_ch=9,  kH=3, kW=3)
        Gen 3: (out_ch=32, in_ch=14, kH=3, kW=3)

    Первые 9 входных каналов — идентичны по смыслу (доска, фигуры, fill, placement).
    Каналы 9-13 явно обнуляем: модель сразу ведёт себя как Gen 2,
    постепенно учится использовать survivability, blob map и dead zones.

    SB3 создаёт три отдельных feature extractor-а:
        features_extractor       (shared при сборе роллаутов)
        pi_features_extractor    (actor)
        vf_features_extractor    (critic)
    Все три имеют Conv1 с разными формами — обрабатываем все.
    """
    # Суффиксы всех Conv1 слоёв которые требуют partial transfer
    CONV1_SUFFIX = ".cnn.0.weight"

    gen2_state = gen2_model.policy.state_dict()
    gen3_state = gen3_model.policy.state_dict()

    transferred, skipped, partial = 0, 0, 0

    for name, gen2_param in gen2_state.items():
        if name not in gen3_state:
            print(f"  [SKIP — not in Gen3] {name}")
            skipped += 1
            continue

        gen3_param = gen3_state[name]

        if name.endswith(CONV1_SUFFIX):
            # Все три Conv1 входных слоя: (32,9,3,3) → (32,14,3,3)
            assert gen2_param.shape == (32, 9, 3, 3), \
                f"Неожиданная форма Gen2 Conv1 '{name}': {gen2_param.shape}"
            assert gen3_param.shape == (32, 14, 3, 3), \
                f"Неожиданная форма Gen3 Conv1 '{name}': {gen3_param.shape}"
            # Каналы 0-8 — веса Gen 2
            gen3_state[name][:, :9, :, :] = gen2_param.clone()
            # Каналы 9-13 — явно нули (PyTorch init использует Kaiming, не нули)
            gen3_state[name][:, 9:, :, :] = 0.0
            print(f"  [PARTIAL] {name}: каналы 0-8 ← Gen2, каналы 9-13 = нули")
            partial += 1

        elif gen2_param.shape == gen3_param.shape:
            gen3_state[name] = gen2_param.clone()
            transferred += 1

        else:
            print(f"  [SKIP — shape mismatch] {name}: Gen2={gen2_param.shape} Gen3={gen3_param.shape}")
            skipped += 1

    gen3_model.policy.load_state_dict(gen3_state)
    print(f"\n  Перенесено: {transferred} слоёв, частично: {partial}, пропущено: {skipped}")


# ================================================================== #
#  Callback: статистика эпизодов
# ================================================================== #

class EpisodeStatsCallback(BaseCallback):
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
#  Основная функция
# ================================================================== #

def transfer_train(
    gen2_model_path: str,
    run_name: str | None,
    total_timesteps: int,
    n_envs: int,
    learning_rate: float,
    max_grad_norm: float,
) -> None:
    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"cnn_gen3_transfer_{ts}"

    save_path    = os.path.join(LOGGING["save_dir"], f"{run_name}_final")
    vecnorm_path = os.path.join(LOGGING["save_dir"], f"{run_name}_vecnormalize.pkl")
    checkpoint_dir = LOGGING["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOGGING["save_dir"], exist_ok=True)

    print(f"\n[transfer_train] run_name={run_name}")
    print(f"[transfer_train] Gen2 source: {gen2_model_path}")
    print(f"[transfer_train] total_timesteps={total_timesteps:,}  n_envs={n_envs}")
    print(f"[transfer_train] lr={learning_rate}  max_grad_norm={max_grad_norm}")

    # ── Окружение Gen 3 (14 каналов) ──────────────────────────────────
    print(f"\n[transfer_train] Создаём {n_envs} параллельных окружений (Gen 3, 14 каналов)...")
    vec_env = SubprocVecEnv([_make_env(rank=i, seed=42) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=CNN_TRAIN.get("gamma", 0.99),
    )

    # ── Создаём Gen 3 модель (та же архитектура, 14 входных каналов) ──
    policy_kwargs = {
        "net_arch": CNN_TRAIN["net_arch"],
        "features_extractor_class":  SmallCNN,
        "features_extractor_kwargs": {
            "features_dim": CNN_TRAIN.get("features_dim", 256),
        },
    }

    gen3_model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=CNN_TRAIN["n_steps"],
        batch_size=CNN_TRAIN["batch_size"],
        n_epochs=CNN_TRAIN["n_epochs"],
        gamma=CNN_TRAIN["gamma"],
        gae_lambda=CNN_TRAIN["gae_lambda"],
        clip_range=CNN_TRAIN["clip_range"],
        ent_coef=CNN_TRAIN["ent_coef"],
        vf_coef=CNN_TRAIN["vf_coef"],
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOGGING["tensorboard_log"],
        verbose=1,
    )

    # ── Загружаем Gen 2 и переносим веса ─────────────────────────────
    print(f"\n[transfer_train] Загружаем Gen2 CNN: {gen2_model_path}")
    gen2_model = MaskablePPO.load(gen2_model_path)

    print("[transfer_train] Переносим веса...")
    transfer_weights(gen2_model, gen3_model)
    del gen2_model  # освобождаем память

    # Санти-чек: первые 9 входных каналов Conv1 должны быть ненулевыми
    conv1_w = gen3_model.policy.features_extractor.cnn[0].weight
    assert conv1_w[:, :9, :, :].abs().sum() > 0, "Conv1 каналы 0-8 остались нулями — что-то пошло не так"
    assert conv1_w[:, 9:, :, :].abs().sum() == 0, "Conv1 каналы 9-13 должны быть нулями после инициализации"
    print("  [OK] Веса перенесены корректно\n")

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks = [
        CheckpointCallback(
            save_freq=max(LOGGING["checkpoint_freq"] // n_envs, 1),
            save_path=checkpoint_dir,
            name_prefix="cnn_transfer",
            verbose=1,
        ),
        EpisodeStatsCallback(log_interval=LOGGING["log_interval"]),
    ]

    # ── Обучение ─────────────────────────────────────────────────────
    log_file_path = os.path.join(
        LOGGING["tensorboard_log"], run_name, "training_console.log"
    )
    with TrainingLogger(log_file_path) as t_logger:
        t_logger.log_header()
        t_logger.log_model_info(gen3_model)
        t_logger.log_params({
            "ARCH":          "CNN (transfer from Gen2)",
            "GEN2_SOURCE":   gen2_model_path,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "total_timesteps": total_timesteps,
            "n_envs":        n_envs,
            "REWARD":        REWARD,
            "ENV":           ENV,
        })
        t_logger.start_capture()

        try:
            print(f"[transfer_train] Начинаем fine-tuning на {total_timesteps:,} шагов...")
            print(f"[transfer_train] Ожидаем быстрый старт: модель уже умеет играть на ~8.7 lines\n")

            gen3_model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=run_name,
                reset_num_timesteps=True,
                progress_bar=True,
            )

            gen3_model.save(save_path)
            print(f"\n[transfer_train] Модель сохранена: {save_path}.zip")

            vec_env.save(vecnorm_path)
            print(f"[transfer_train] VecNormalize сохранён: {vecnorm_path}")

        finally:
            t_logger.stop_capture()

    vec_env.close()


# ================================================================== #
#  CLI
# ================================================================== #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer learning: Gen 2 CNN → Gen 3 CNN (14 каналов)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gen2-model", type=str, default=GEN2_MODEL_DEFAULT,
        help=f"Путь к Gen 2 .zip модели (default: {GEN2_MODEL_DEFAULT})"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Имя запуска (default: cnn_gen3_transfer_<timestamp>)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000,
        help="Число шагов fine-tuning (default: 5_000_000)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=CNN_TRAIN["n_envs"],
        help=f"Число параллельных окружений (default: {CNN_TRAIN['n_envs']})"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate для fine-tuning (default: 5e-5, меньше чем 7e-5 при обучении с нуля)"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.3,
        help="Клиппинг градиентов (default: 0.3, меньше чем 0.5 при обучении с нуля)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    transfer_train(
        gen2_model_path=args.gen2_model,
        run_name=args.run_name,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
    )
