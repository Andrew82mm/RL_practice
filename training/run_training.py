"""
run_training.py — единая точка запуска обучения (MLP / CNN / Spatial CNN).

Использование:
    # Обычное обучение CNN с нуля
    python training/run_training.py --arch cnn --run-name cnn_16x16_v1

    # CNN с Behavior Cloning warm-start
    python training/bc_train.py                                    # фаза 1
    python training/run_training.py --arch cnn --bc-pretrained ./models/bc_pretrained.zip

    # Spatial CNN (пространственная голова актора)
    python training/run_training.py --arch spatial --run-name spatial_16x16_v1

    # MLP
    python training/run_training.py --arch mlp --run-name mlp_16x16_v1

    # Продолжить с чекпоинта
    python training/run_training.py --arch cnn --resume ./models/checkpoints/cnn_puzzle_5000000_steps.zip
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from config import REWARD, ENV, LOGGING, MLP_TRAIN, CNN_TRAIN, SPATIAL_CNN_TRAIN, VIT_TRAIN
from block_puzzle_env.environment import BlockPuzzleEnv
from utils.logger import TrainingLogger
from utils.cnn_extractor import HierarchicalCNN, SpatialCNNExtractor, SpatialMaskableActorCriticPolicy
from utils.vit_extractor import SmallViT


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
#  Callback: логирует ep-метрики
# ================================================================== #

class EpisodeStatsCallback(BaseCallback):
    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self._log_interval = log_interval
        self._lines:   list[int] = []
        self._perfect: list[int] = []
        self._placed:  list[int] = []
        self._ev_warned = False

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

        if (not self._ev_warned
                and self.num_timesteps > 200_000
                and hasattr(self.model, "logger")):
            ev = self.model.logger.name_to_value.get("train/explained_variance", None)
            if ev is not None and ev < 0.1:
                print(
                    f"\n[WARNING] explained_variance={ev:.3f} после {self.num_timesteps:,} шагов. "
                    f"Критик не обучается — проверь нормализацию наград.\n"
                )
                self._ev_warned = True

        return True


# ================================================================== #
#  policy_kwargs
# ================================================================== #

def _build_policy_kwargs(arch: str, train_cfg: dict) -> dict:
    net_arch = train_cfg.get("net_arch", {"pi": [256, 256], "vf": [512, 512]})

    if arch == "mlp":
        return {"net_arch": net_arch}

    if arch == "cnn":
        return {
            "net_arch": net_arch,
            "features_extractor_class":  HierarchicalCNN,
            "features_extractor_kwargs": {
                "features_dim": train_cfg.get("features_dim", 512),
            },
        }

    if arch == "vit":
        return {
            "net_arch": net_arch,
            "features_extractor_class":  SmallViT,
            "features_extractor_kwargs": {
                "features_dim": train_cfg.get("features_dim", 512),
                "embed_dim":    train_cfg.get("embed_dim", 256),
                "n_heads":      train_cfg.get("n_heads", 4),
                "n_layers":     train_cfg.get("n_layers", 2),
                "ffn_dim":      train_cfg.get("ffn_dim", 512),
            },
        }

    if arch == "spatial":
        # Для spatial: pi-экстрактор = SpatialCNNExtractor, vf = HierarchicalCNN.
        # SB3 с dict net_arch создаёт раздельные pi/vf экстракторы.
        # features_extractor_class используется для обоих по умолчанию —
        # для spatial нужна особая инициализация policy (см. SpatialMaskableActorCriticPolicy).
        return {
            "net_arch":                  {"pi": [], "vf": [512, 256]},
            "features_extractor_class":  SpatialCNNExtractor,
            "features_extractor_kwargs": {},
        }

    raise ValueError(f"Неизвестная архитектура: '{arch}'")


# ================================================================== #
#  Загрузка BC-весов в свежую PPO-модель
# ================================================================== #

def _load_bc_weights(model: MaskablePPO, bc_path: str) -> None:
    """
    Переносит веса policy (актор + feature extractor) из BC-модели в PPO.
    Веса критика остаются случайными (он не обучался в BC).
    """
    print(f"[run_training] Загружаем BC-веса из: {bc_path}")
    bc_model = MaskablePPO.load(bc_path)

    bc_state    = bc_model.policy.state_dict()
    model_state = model.policy.state_dict()

    transferred, skipped = 0, 0
    for name, param in bc_state.items():
        # Копируем только pi-пути (актор + его feature extractor)
        # Пропускаем vf_features_extractor и value_net
        if "vf_features_extractor" in name or "value_net" in name:
            skipped += 1
            continue
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param.clone()
            transferred += 1
        else:
            skipped += 1

    model.policy.load_state_dict(model_state)
    del bc_model
    print(f"[run_training] BC-веса перенесены: {transferred} слоёв, пропущено: {skipped}")


# ================================================================== #
#  Основная функция
# ================================================================== #

def train(
    arch: str,
    run_name: str | None = None,
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    resume_path: str | None = None,
    bc_pretrained: str | None = None,
    normalize_reward: bool = True,
) -> None:
    if arch == "mlp":
        train_cfg = dict(MLP_TRAIN)
        policy_cls = "MlpPolicy"
    elif arch == "cnn":
        train_cfg = dict(CNN_TRAIN)
        policy_cls = "MlpPolicy"
    elif arch == "vit":
        train_cfg = dict(VIT_TRAIN)
        policy_cls = "MlpPolicy"
    elif arch == "spatial":
        train_cfg = dict(SPATIAL_CNN_TRAIN)
        policy_cls = SpatialMaskableActorCriticPolicy
    else:
        raise ValueError(f"Неизвестная архитектура: '{arch}'. Допустимо: mlp, cnn, vit, spatial.")

    if total_timesteps is not None:
        train_cfg["total_timesteps"] = total_timesteps
    if n_envs is not None:
        train_cfg["n_envs"] = n_envs

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{arch}_{timestamp}"

    save_path      = os.path.join(LOGGING["save_dir"], f"{run_name}_final")
    vecnorm_path   = os.path.join(LOGGING["save_dir"], f"{run_name}_vecnormalize.pkl")
    checkpoint_dir = LOGGING["checkpoint_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOGGING["save_dir"], exist_ok=True)

    n_envs_actual = train_cfg["n_envs"]
    total_ts      = train_cfg["total_timesteps"]

    print(f"[run_training] arch={arch.upper()}  run_name={run_name}")
    print(f"[run_training] board_size={ENV['board_size']}×{ENV['board_size']}")
    print(f"[run_training] normalize_reward={normalize_reward}")
    print(f"[run_training] Создаём {n_envs_actual} параллельных окружений...")

    vec_env = SubprocVecEnv([_make_env(rank=i, seed=42) for i in range(n_envs_actual)])
    vec_env = VecMonitor(vec_env)

    if normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=train_cfg.get("gamma", 0.99),
        )

    policy_kwargs = _build_policy_kwargs(arch, train_cfg)

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
            policy=policy_cls,
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

    # BC warm-start: грузим веса актора из BC-модели
    if bc_pretrained and not resume_path:
        _load_bc_weights(model, bc_pretrained)

    callbacks = [
        CheckpointCallback(
            save_freq=max(LOGGING["checkpoint_freq"] // n_envs_actual, 1),
            save_path=checkpoint_dir,
            name_prefix=f"{arch}_puzzle",
            verbose=1,
        ),
        EpisodeStatsCallback(log_interval=LOGGING["log_interval"]),
    ]

    log_file_path = os.path.join(
        LOGGING["tensorboard_log"], run_name, "training_console.log"
    )

    with TrainingLogger(log_file_path) as t_logger:
        t_logger.log_header()
        t_logger.log_model_info(model)
        t_logger.log_params({
            "ARCH":             arch.upper(),
            "BC_PRETRAINED":    bc_pretrained,
            "normalize_reward": normalize_reward,
            "TRAIN":            {k: str(v) for k, v in train_cfg.items()},
            "REWARD":           REWARD,
            "ENV":              ENV,
        })
        t_logger.start_capture()

        try:
            print(f"[run_training] Начинаем обучение на {total_ts:,} шагов...")
            if bc_pretrained:
                print("[run_training] Старт с BC warm-start: ожидаем ~40+ lines с первых шагов")

            model.learn(
                total_timesteps=total_ts,
                callback=callbacks,
                tb_log_name=run_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=True,
            )

            model.save(save_path)
            print(f"[run_training] Модель сохранена: {save_path}.zip")

            if normalize_reward and isinstance(vec_env, VecNormalize):
                vec_env.save(vecnorm_path)
                print(f"[run_training] VecNormalize сохранён: {vecnorm_path}")

        finally:
            t_logger.stop_capture()

    vec_env.close()


# ================================================================== #
#  CLI
# ================================================================== #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block Puzzle RL — обучение MLP / CNN / Spatial CNN агента",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--arch", type=str, choices=["mlp", "cnn", "spatial"], default="cnn",
        help="Архитектура: mlp, cnn, spatial (default: cnn)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Имя запуска. Если не задано — <arch>_<timestamp>"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Переопределить total_timesteps из конфига"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help="Переопределить число параллельных окружений"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Путь к .zip чекпоинту для продолжения обучения"
    )
    parser.add_argument(
        "--bc-pretrained", type=str, default=None,
        help="Путь к BC-модели для warm-start (из bc_train.py)"
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Отключить VecNormalize (не рекомендуется)"
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
        bc_pretrained=args.bc_pretrained,
        normalize_reward=not args.no_normalize,
    )
