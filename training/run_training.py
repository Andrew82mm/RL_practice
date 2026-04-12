"""
run_training.py - единая точка запуска обучения для MLP и CNN архитектур.

КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v2: добавлен VecNormalize для нормализации наград.
Проблема предыдущей версии: explained_variance = 0.03 на 5млн (норма > 0.8).
Причина: combo-награды создают огромную дисперсию (от -3 до +500+),
критик не может стабильно предсказывать возвраты => градиент политики - шум.
Решение: VecNormalize нормализует награды через бегущее среднее/std,
что возвращает все сигналы в диапазон ~[-5, +5].

Использование:
    python run_training.py --arch mlp --run-name mlp_v2
    python run_training.py --arch cnn --run-name cnn_v2
    python run_training.py --arch mlp --run-name mlp_v2 --timesteps 2000000 --n-envs 4
    python run_training.py --arch mlp --run-name mlp_v2 \\
        --resume ./models/checkpoints/mlp_puzzle_1000000_steps.zip

Структура файлов:
    ./runs/<run_name>/                     — TensorBoard + training_console.log
    ./models/checkpoints/                  — чекпоинты каждые checkpoint_freq шагов
    ./models/<run_name>_final.zip          — финальная модель
    ./models/<run_name>_vecnormalize.pkl   — статистика нормализации (нужна для eval!)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from config import REWARD, ENV, LOGGING, MLP_TRAIN, CNN_TRAIN
from block_puzzle_env.environment import BlockPuzzleEnv
from utils.logger import TrainingLogger


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
#  Callback: логирует ep-метрики + explained_variance предупреждение
# ================================================================== #

class EpisodeStatsCallback(BaseCallback):
    """
    Логирует ep_lines_cleared, ep_pieces_placed, ep_perfect_clears
    только при завершении эпизода (done=True).
    Дополнительно предупреждает если explained_variance < 0.3 —
    признак того что критик не обучается.
    """
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

        # Предупреждение о плохом критике (проверяем раз, через 100k шагов)
        if (not self._ev_warned
                and self.num_timesteps > 100_000
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

def _build_policy_kwargs(train_cfg: dict) -> dict:
    arch    = train_cfg["features_extractor"]
    net_arch = train_cfg.get("net_arch", [256, 256])

    if arch == "flatten":
        return {"net_arch": net_arch}

    if arch == "cnn":
        from utils.cnn_extractor import SmallCNN
        return {
            "net_arch": net_arch,
            "features_extractor_class":  SmallCNN,
            "features_extractor_kwargs": {
                "features_dim": train_cfg.get("features_dim", 256),
            },
        }

    raise ValueError(f"Неизвестный features_extractor: '{arch}'")


# ================================================================== #
#  Основная функция
# ================================================================== #

def train(
    arch: str,
    run_name: str | None = None,
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    resume_path: str | None = None,
    normalize_reward: bool = True,
) -> None:
    """
    Args:
        arch:             "mlp" или "cnn"
        run_name:         имя запуска. Если None — генерируется автоматически.
        total_timesteps:  переопределяет значение из конфига.
        n_envs:           переопределяет значение из конфига.
        resume_path:      путь к .zip чекпоинту для продолжения обучения.
        normalize_reward: включить VecNormalize для нормализации наград
                          (реккомендуется, решает проблему explained_variance~0).
    """
    if arch == "mlp":
        train_cfg = dict(MLP_TRAIN)
    elif arch == "cnn":
        train_cfg = dict(CNN_TRAIN)
    else:
        raise ValueError(f"Неизвестная архитектура: '{arch}'. Допустимо: 'mlp', 'cnn'.")

    if total_timesteps is not None:
        train_cfg["total_timesteps"] = total_timesteps
    if n_envs is not None:
        train_cfg["n_envs"] = n_envs

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{arch}_{timestamp}"

    save_path       = os.path.join(LOGGING["save_dir"], f"{run_name}_final")
    vecnorm_path    = os.path.join(LOGGING["save_dir"], f"{run_name}_vecnormalize.pkl")
    checkpoint_dir  = LOGGING["checkpoint_dir"]
    checkpoint_prefix = f"{arch}_puzzle"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOGGING["save_dir"], exist_ok=True)

    n_envs_actual = train_cfg["n_envs"]
    total_ts      = train_cfg["total_timesteps"]

    print(f"[run_training] arch={arch.upper()}  run_name={run_name}")
    print(f"[run_training] normalize_reward={normalize_reward}")
    print(f"[run_training] Создаём {n_envs_actual} параллельных окружений...")

    # ── Векторное окружение ───────────────────────────────────────────
    vec_env = SubprocVecEnv([_make_env(rank=i, seed=42) for i in range(n_envs_actual)])
    vec_env = VecMonitor(vec_env)

    if normalize_reward:
        # norm_obs=False — наблюдения уже в [0,1], нормировать не нужно
        # norm_reward=True — нормируем награды через бегущее std
        # clip_reward=10.0 — отсекаем экстремальные выбросы (perfect_clear=15 → ~3 std)
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=train_cfg.get("gamma", 0.99),
        )

    # ── Модель ───────────────────────────────────────────────────────
    policy_kwargs = _build_policy_kwargs(train_cfg)

    if resume_path:
        print(f"[run_training] Загружаем чекпоинт: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=vec_env,
            tensorboard_log=LOGGING["tensorboard_log"],
        )
        # Восстановить VecNormalize статистику если она была сохранена
        resume_vecnorm = resume_path.replace("_steps.zip", "_vecnormalize.pkl")
        if normalize_reward and os.path.exists(resume_vecnorm):
            vec_env = VecNormalize.load(resume_vecnorm, vec_env)
            print(f"[run_training] VecNormalize статистика восстановлена: {resume_vecnorm}")
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

    # ── Callback: сохранение чекпоинтов + VecNormalize статистики ────
    class CheckpointWithVecNorm(CheckpointCallback):
        """
        Расширяет стандартный CheckpointCallback: вместе с моделью
        сохраняет текущую статистику VecNormalize (running mean/std наград).
        Без этого при resume обучение стартует с нуля по статистике,
        что ломает нормализацию в первые итерации после возобновления.
        Но с ним генерируется абсурдно большое количество .pkl файлов
        Примерно 57 тысяч за ~500к шагов обучения mlp
        """
        def __init__(self, vec_env_ref, vecnorm_save_dir, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Убрал сохранение статистики нормализации до лучших времен
            #self._vec_env_ref   = vec_env_ref
            #self._vecnorm_dir   = vecnorm_save_dir

        def _on_step(self) -> bool:
            result = super()._on_step()
            #if normalize_reward and isinstance(self._vec_env_ref, VecNormalize):
            #   vecnorm_ckpt = os.path.join(
            #        self._vecnorm_dir,
            #        f"{checkpoint_prefix}_{self.num_timesteps}_vecnormalize.pkl"
            #    )
            #    self._vec_env_ref.save(vecnorm_ckpt)
            return result

    callbacks = [
        CheckpointWithVecNorm(
            vec_env_ref=vec_env,
            vecnorm_save_dir=checkpoint_dir,
            save_freq=max(LOGGING["checkpoint_freq"] // n_envs_actual, 1),
            save_path=checkpoint_dir,
            name_prefix=checkpoint_prefix,
            verbose=1,
        ),
        EpisodeStatsCallback(log_interval=LOGGING["log_interval"]),
    ]

    # ── Логгер ───────────────────────────────────────────────────────
    log_file_path = os.path.join(
        LOGGING["tensorboard_log"], run_name, "training_console.log"
    )

    with TrainingLogger(log_file_path) as t_logger:
        t_logger.log_header()
        t_logger.log_model_info(model)
        t_logger.log_params({
            "ARCH":             arch.upper(),
            "normalize_reward": normalize_reward,
            "TRAIN":            train_cfg,
            "REWARD":           REWARD,
            "ENV":              ENV,
        })
        t_logger.start_capture()

        try:
            print(f"[run_training] Начинаем обучение на {total_ts:,} шагов...")
            print(f"[run_training] Ожидаемый explained_variance > 0.5 после 500k шагов\n")

            model.learn(
                total_timesteps=total_ts,
                callback=callbacks,
                tb_log_name=run_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=True,
            )

            # Сохранить модель
            model.save(save_path)
            print(f"[run_training] Модель сохранена: {save_path}.zip")

            # Сохранить VecNormalize статистику (нужна для evaluate.py!)
            if normalize_reward and isinstance(vec_env, VecNormalize):
                vec_env.save(vecnorm_path)
                print(f"[run_training] VecNormalize сохранён: {vecnorm_path}")
                print(f"[run_training] При оценке указывай --vecnormalize {vecnorm_path}")

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
        "--arch", type=str, choices=["mlp", "cnn"], default="mlp",
        help="Архитектура: mlp или cnn (default: mlp)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None, metavar="NAME",
        help="Имя запуска. Если не задано — <arch>_<timestamp>"
    )   
    parser.add_argument(
        "--timesteps", type=int, default=None, metavar="N",
        help="Переопределить total_timesteps из конфига"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None, metavar="N",
        help="Переопределить число параллельных окружений"
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="PATH",
        help="Путь к .zip чекпоинту для продолжения обучения"
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
        normalize_reward=not args.no_normalize,
    )
