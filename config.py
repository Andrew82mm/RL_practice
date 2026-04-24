
# config.py  (v5 — 16×16 board, spatial policy, behaviour cloning support)
#
# Изменения v5 (переход 8×8 → 16×16):
#   board_size:       8   → 16
#   max_steps:        2000 → 8000    (эпизоды длиннее в ~10×)
#   combo_multiplier: 1.0 → 1.5     (комбо-бонус включён: 2 линии = 9 вместо 6)
#   cell_potential:   0.3 → 0.1     (на 16×16 row_fill редко близок к 1 сразу)
#   game_over_per_cell: -0.1 → -0.03 (256 клеток × -0.03 = -7.7 вместо 64 × -0.1 = -6.4)
#   round_complete:   0.5 → 0.3     (раундов гораздо больше, накапливается)
#   n_steps:          512  → 1024   (длиннее эпизоды → длиннее trajectory для GAE)
#   batch_size:       256  → 512
#   CNN lr:           7e-5 → 5e-5   (16×16 CNN имеет больше параметров)
#   features_dim:     256  → 512    (нужна бо́льшая ёмкость для 16×16)
#   total_timesteps:  5M   → 30M    (16×16 сходится дольше)
#   learning_rate:    fixed → linear schedule (7e-5 → 1e-5)

import numpy as np


def _linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    """Linear LR schedule: initial_lr → final_lr по мере обучения."""
    def func(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return func


# ----------------------------------------------------------
# Награды и штрафы (reward shaping)
# ----------------------------------------------------------
REWARD = {
    # За успешное размещение фигуры (базовый плотный сигнал).
    "place_piece": 0.2,
    # За очищенную линию
    "line_cleared": 3.0,
    # Комбо-множитель: управляет бонусом за очистку нескольких линий за раз.
    # 1.5 → 1 линия=3, 2=9, 3=20.25, 4=40.5 — квадратичный рост стимулирует combo.
    # На 16×16 одновременная очистка нескольких линий встречается заметно чаще.
    "combo_multiplier": 1.5,
    # Perfect Clear
    "perfect_clear": 15.0,
    # Потенциальная награда за каждую клетку фигуры.
    # Уменьшена 0.3 → 0.1: на 16×16 row_fill редко близок к 1 сразу после хода,
    # основной сигнал идёт от line_cleared, а не от cell_potential.
    "cell_potential": 0.1,
    # Защитный штраф (не должен срабатывать с action mask)
    "invalid_move": -2.0,
    # Штраф за конец игры
    "game_over": -5.0,
    # Штраф за занятые клетки при game over.
    # -0.1 → -0.03: на 16×16 до 256 занятых клеток (256 × -0.03 = -7.7),
    # на 8×8 было до 64 клеток (64 × -0.1 = -6.4) — масштабируем аналогично.
    "game_over_per_cell": -0.03,
    # Бонус за успешное завершение раунда.
    # -0.5 → 0.3: раундов на 16×16 гораздо больше (длиннее эпизоды),
    # чтобы не доминировало над line_cleared.
    "round_complete": 0.3,
}

# ----------------------------------------------------------
# Параметры окружения
# ----------------------------------------------------------
ENV = {
    "board_size": 16,
    "pieces_per_round": 3,
    # Максимальное число шагов в эпизоде.
    # 16×16 эпизоды в ~10× длиннее 8×8, ставим 8000.
    "max_steps": 8000,
}

# ----------------------------------------------------------
# Логирование
# ----------------------------------------------------------
LOGGING = {
    "tensorboard_log": "./runs/",
    "run_name": None,
    "save_dir": "./models/",
    "checkpoint_freq": 500_000,
    "checkpoint_dir": "./models/checkpoints/",
    "log_interval": 10,
}

# ----------------------------------------------------------
# Общие PPO-гиперпараметры
# ----------------------------------------------------------
_PPO_BASE = {
    "n_envs":          16,
    "total_timesteps": 30_000_000,

    # Linear schedule: 7e-5 → 1e-5.
    # MLP будет переопределять на свой schedule.
    "learning_rate":   _linear_schedule(7e-5, 1e-5),

    # 1024 шагов × 16 envs = 16 384 переходов до обновления.
    # Длиннее эпизоды на 16×16 → нужна длиннее trajectory для точного GAE.
    "n_steps":         1024,

    "batch_size":      512,
    "n_epochs":        4,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.005,
    "vf_coef":         1.0,
    "max_grad_norm":   0.5,
}

# ----------------------------------------------------------
# MLP — политика MlpPolicy + FlattenExtractor
#
# Вход: flatten(15 × 16 × 16) = 3840 значений.
# Бо́льшая сеть нужна для 16×16 — видит 4× больше клеток.
# ----------------------------------------------------------
MLP_TRAIN = {
    **_PPO_BASE,
    "learning_rate":  _linear_schedule(1e-4, 2e-5),
    "policy":         "MlpPolicy",
    "net_arch":       {"pi": [512, 512], "vf": [1024, 1024]},
    "features_extractor": "flatten",
}

# ----------------------------------------------------------
# CNN — политика с кастомным HierarchicalCNN экстрактором
#
# Архитектура HierarchicalCNN (вход C × 16 × 16):
#   Conv(C→32,   3×3, pad=1)          → ReLU → (32, 16, 16)
#   Conv(32→64,  3×3, stride=2, pad=1) → ReLU → (64,  8,  8)  [↓2]
#   Conv(64→128, 3×3, pad=1)           → ReLU → (128, 8,  8)
#   Conv(128→128,3×3, stride=2, pad=1) → ReLU → (128, 4,  4)  [↓2]
#   Flatten → Linear(2048 → features_dim) + ReLU
#
# features_dim=512: даёт 4× больше ёмкости чем Gen3 (256).
# ----------------------------------------------------------
CNN_TRAIN = {
    **_PPO_BASE,
    "learning_rate":     _linear_schedule(5e-5, 5e-6),
    "policy":            "MlpPolicy",
    "features_dim":      512,
    "net_arch":          {"pi": [512, 256], "vf": [1024, 512]},
    "features_extractor": "cnn",
}

# ----------------------------------------------------------
# Spatial CNN — CNN с пространственной головой политики
#
# Актор использует SpatialCNNExtractor: лёгкий encoder-decoder
# выдаёт логиты прямо в пространстве (3 × 16 × 16 = 768).
# Критик использует HierarchicalCNN → features_dim=512.
# ----------------------------------------------------------
SPATIAL_CNN_TRAIN = {
    **_PPO_BASE,
    "learning_rate":      _linear_schedule(5e-5, 5e-6),
    "policy":             "SpatialMaskableActorCriticPolicy",
    "features_dim":       512,
    "features_extractor": "spatial_cnn",
}

# ----------------------------------------------------------
# Behavior Cloning (supervised pre-training от эвристики)
# ----------------------------------------------------------
BC_CONFIG = {
    # Сколько (obs, action) пар собрать от HeuristicAgent
    "n_samples": 500_000,
    # Файл для сохранения датасета
    "dataset_path": "./models/bc_dataset.npz",
    # Число эпох supervised обучения
    "n_epochs": 50,
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    # После BC: путь к сохранённым весам политики (передаётся в PPO как warm-start)
    "pretrained_path": "./models/bc_pretrained.zip",
}
