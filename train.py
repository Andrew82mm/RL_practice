import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from config import REWARD, ENV, LOGGING, TRAIN
from block_puzzle_env.environment import BlockPuzzleEnv
from logger import TrainingLogger

# ------------------------------------------------------------------ #
#  Вспомогательные функции
# ------------------------------------------------------------------ #

def make_mask_fn(env: gym.Env):
    """Возвращает маску действий для ActionMasker."""
    return env.action_masks()


def make_env(rank: int = 0, seed: int = 0):
    """Фабрика окружений для SubprocVecEnv."""
    def _init():
        env = BlockPuzzleEnv()
        env = ActionMasker(env, make_mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init


# ------------------------------------------------------------------ #
#  TensorBoard callback — логирует метрики эпизода
# ------------------------------------------------------------------ #

class EpisodeStatsCallback(BaseCallback):
    """
    Собирает ep_lines_cleared, ep_perfect_clears, ep_pieces_placed из info и логирует средние значения в TensorBoard.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._lines: list[int] = []
        self._perfect: list[int] = []
        self._placed: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # VecEnv оборачивает финальные info в "final_info" при done=True
            ep_info = info.get("final_info") or info
            if "ep_lines_cleared" in ep_info:
                self._lines.append(ep_info["ep_lines_cleared"])
                self._perfect.append(ep_info["ep_perfect_clears"])
                self._placed.append(ep_info["ep_pieces_placed"])

        log_interval = LOGGING.get("log_interval", 10)
        if len(self._lines) >= log_interval:
            self.logger.record("episode/lines_cleared_mean",  np.mean(self._lines))
            self.logger.record("episode/perfect_clears_mean", np.mean(self._perfect))
            self.logger.record("episode/pieces_placed_mean",  np.mean(self._placed))
            self._lines.clear()
            self._perfect.clear()
            self._placed.clear()

        return True


# ------------------------------------------------------------------ #
#  Основная функция обучения
# ------------------------------------------------------------------ #

def train():
    os.makedirs(LOGGING["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(LOGGING["save_path"]), exist_ok=True)

    n_envs = TRAIN.get("n_envs", 8)

    # Выводим сообщение ДО создания логгера, чтобы оно не попало в файл
    print(f"[train] Создаём {n_envs} параллельных окружений...")
    
    vec_env = SubprocVecEnv([make_env(rank=i, seed=42) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        net_arch=TRAIN.get("net_arch", [256, 256]),
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=TRAIN.get("learning_rate", 3e-4),
        n_steps=TRAIN.get("n_steps", 2048),
        batch_size=TRAIN.get("batch_size", 512),
        n_epochs=TRAIN.get("n_epochs", 10),
        gamma=TRAIN.get("gamma", 0.99),
        gae_lambda=TRAIN.get("gae_lambda", 0.95),
        clip_range=TRAIN.get("clip_range", 0.2),
        ent_coef=TRAIN.get("ent_coef", 0.03),
        vf_coef=TRAIN.get("vf_coef", 0.5),
        max_grad_norm=TRAIN.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOGGING["tensorboard_log"],
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(LOGGING["checkpoint_freq"] // n_envs, 1),
            save_path=LOGGING["checkpoint_dir"],
            name_prefix="block_puzzle",
            verbose=1,
        ),
        EpisodeStatsCallback(),
    ]

    total_timesteps = TRAIN.get("total_timesteps", 5_000_000)
    
    # ---------------------------------------------------------- #
    # НАСТРОЙКА ЛОГГИРОВАНИЯ
    # ---------------------------------------------------------- #
    log_file_path = os.path.join(LOGGING["tensorboard_log"], LOGGING["run_name"], "training_console.log")
    
    with TrainingLogger(log_file_path) as t_logger:
        # 1. Шапка
        t_logger.log_header()
        
        # 2. Информация о модели (Алгоритм + Архитектура)
        t_logger.log_model_info(model)
        
        # 3. Параметры конфигурации
        t_logger.log_params({
            "TRAIN": TRAIN,
            "REWARD": REWARD,
            "ENV": ENV
        })
        
        # 4. Перехват вывода (stdout + stderr)
        t_logger.start_capture()
        
        try:
            # Это сообщение теперь попадет в лог
            print(f"[train] Начинаем обучение на {total_timesteps:,} шагов...")
            
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=LOGGING["run_name"],
                reset_num_timesteps=True,
                progress_bar=True,
            )
            model.save(LOGGING["save_path"])
            print(f"[train] Модель сохранена: {LOGGING['save_path']}.zip")
        finally:
            t_logger.stop_capture()
            
    vec_env.close()


# ------------------------------------------------------------------ #
#  Точка входа
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    train()