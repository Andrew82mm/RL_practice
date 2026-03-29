"""
cnn_extractor.py — кастомный CNN экстрактор признаков для поля 8×8.

Проблема со стандартным CnnPolicy в SB3:
    NatureCNN (используемый по умолчанию в CnnPolicy) содержит свёрточные
    слои со страйдами 4 и 2, которые рассчитаны на вход >= 36×36 пикселей
    (Atari). На поле 8×8 spatial размерность схлопывается в отрицательные
    значения и возникает RuntimeError ещё до первого шага обучения.

Решение — SmallCNN:
    Кастомный экстрактор наследует BaseFeaturesExtractor из SB3 и
    передаётся через policy_kwargs["features_extractor_class"].
    Используем страйд=1 и padding=1, чтобы сохранить spatial размерность
    8×8 через все свёрточные слои.

Архитектура (вход: FloatTensor[B, 4, 8, 8]):

    ┌──────────────────────────────────────────────┐
    │  Conv2d(4→32,  3×3, pad=1) + BN + ReLU       │  → (B, 32, 8, 8)
    │  Conv2d(32→64, 3×3, pad=1) + BN + ReLU       │  → (B, 64, 8, 8)
    │  Conv2d(64→64, 3×3, pad=1) + BN + ReLU       │  → (B, 64, 8, 8)
    │  Flatten                                      │  → (B, 4096)
    │  Linear(4096 → features_dim) + ReLU           │  → (B, features_dim)
    └──────────────────────────────────────────────┘

    features_dim по умолчанию = 256 (задаётся в CNN_TRAIN["features_dim"]).
    Далее SB3 добавляет MLP-головы actor/critic из net_arch поверх эмбеддинга.

BatchNorm:
    Добавлен после каждой свёртки для стабилизации обучения.
    BN хорошо работает с большими batch_size (512 в нашем конфиге).
    При batch_size < 8 можно заменить на LayerNorm или убрать совсем.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    """
    Кастомный CNN экстрактор для наблюдений формата (C, H, W) с малым H, W.

    Args:
        observation_space: пространство наблюдений gym.spaces.Box.
        features_dim:      размерность выходного эмбеддинга (default: 256).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 4 канала

        # --- Свёрточный блок ---
        # Все слои: stride=1, padding=1 → spatial размерность не меняется (8→8)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Flatten(),  # (B, 64*H*W)
        )

        # Вычисляем плоский размер одним форвардом через dummy-тензор
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            flat_size = self.cnn(sample).shape[1]

        # --- Проекционный слой ---
        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
