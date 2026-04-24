"""
cnn_extractor.py — CNN экстракторы и Spatial Policy для Block Puzzle 16×16.

Три класса:
  HierarchicalCNN  — основной экстрактор для vf-головы и MLP-политики.
                     Иерархические свёртки (stride=2) дают реальный receptive field
                     на 16×16. Выход: (B, features_dim) — flat вектор.

  SpatialCNNExtractor — лёгкий encoder-decoder для актор-головы.
                         Выдаёт (B, 3 * board_size²) — пространственные логиты напрямую.
                         Не требует отдельного Linear actor-head.

  SpatialMaskableActorCriticPolicy — кастомный MaskableActorCriticPolicy:
                         action_net = Identity (логиты уже готовы после SpatialCNNExtractor),
                         vf использует HierarchicalCNN → MLP.

Архитектуры:

  HierarchicalCNN (вход C × 16 × 16, features_dim=512):
    Conv(C→32,   3×3, pad=1)           + ReLU  → (32, 16, 16)
    Conv(32→64,  3×3, stride=2, pad=1) + ReLU  → (64,  8,  8)
    Conv(64→128, 3×3, pad=1)           + ReLU  → (128, 8,  8)
    Conv(128→128,3×3, stride=2, pad=1) + ReLU  → (128, 4,  4)
    Flatten → Linear(2048 → features_dim) + ReLU

  SpatialCNNExtractor (вход C × board_size × board_size):
    Conv(C→32, 3×3, pad=1)            + ReLU  → (32, 16, 16)
    Conv(32→64, 3×3, stride=2, pad=1) + ReLU  → (64,  8,  8)
    ConvTranspose(64→32, 2×2, stride=2) + ReLU → (32, 16, 16)
    Conv(32→3, 1×1)                           → (3,  16, 16)
    Flatten                                    → (3 * 16 * 16 = 768)
  Это encoder-decoder: сжимаем → разворачиваем → 3 канала-слота → логиты.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    _HAS_SB3_CONTRIB = True
except ImportError:
    _HAS_SB3_CONTRIB = False


# ===========================================================================
# HierarchicalCNN  (основной экстрактор: critic + MLP-политика)
# ===========================================================================

class HierarchicalCNN(BaseFeaturesExtractor):
    """
    Иерархический CNN для поля 16×16.
    stride=2 на двух слоях даёт receptive field 9×9 — реальное пространственное
    понимание, в отличие от 3×3-only SmallCNN на 8×8.

    Args:
        observation_space: gym.spaces.Box, shape (C, H, W).
        features_dim:      размерность выходного эмбеддинга (default: 512).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Слой 1: локальные признаки, spatial = H×W
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Слой 2: downsampling ×2 → H/2 × W/2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Слой 3: глубокие признаки на уменьшенной карте
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Слой 4: downsampling ×2 → H/4 × W/4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            flat_size = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# обратная совместимость: SmallCNN теперь = HierarchicalCNN
SmallCNN = HierarchicalCNN


# ===========================================================================
# SpatialCNNExtractor  (лёгкий encoder-decoder для актор-головы)
# ===========================================================================

class SpatialCNNExtractor(BaseFeaturesExtractor):
    """
    Encoder-decoder CNN: выдаёт пространственные логиты (n_slots × H × W).

    Используется как pi_features_extractor в SpatialMaskableActorCriticPolicy.
    Выход features_dim = n_slots * H * W (например 3 * 16 * 16 = 768).
    Эти значения — прямые логиты действий, action_net в политике = Identity.

    Encoder-decoder сохраняет позиционный смысл:
    логит (slot_i, y, x) отражает «насколько хорошо ставить фигуру i в (x, y)».
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        n_slots: int = 3,
    ) -> None:
        _, H, W = observation_space.shape
        features_dim = n_slots * H * W
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self._n_slots = n_slots
        self._H = H
        self._W = W

        # Encoder: (C, H, W) → (64, H/2, W/2)
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder: (64, H/2, W/2) → (32, H, W) → (n_slots, H, W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, n_slots, kernel_size=1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.encoder(observations)
        x = self.decoder(x)                        # (B, n_slots, H, W)
        return x.reshape(x.shape[0], -1)           # (B, n_slots * H * W)


# ===========================================================================
# SpatialMaskableActorCriticPolicy
# ===========================================================================

if _HAS_SB3_CONTRIB:

    class SpatialMaskableActorCriticPolicy(MaskableActorCriticPolicy):
        """
        Кастомная политика для Block Puzzle:
          - Актор: SpatialCNNExtractor → Identity (логиты = spatial map)
          - Критик: HierarchicalCNN → MLP → value

        Зачем:
          Стандартный action_net = Linear(features_dim, n_actions) теряет
          пространственный смысл (shuffled logits). Здесь логиты уже упорядочены
          как (slot, y, x) — агент учится «куда поставить», а не «какой индекс».

        Настройка через policy_kwargs:
          features_dim:      размерность vf-экстрактора (default: 512)
          pi_features_dim:   n_slots * H * W (вычисляется автоматически, не задаётся)
          net_arch:          только vf-MLP; pi-MLP не используется → {}
        """

        def _build(self, lr_schedule) -> None:
            # SB3 создаёт отдельные pi/vf экстракторы когда net_arch — dict
            # Здесь: pi_features_extractor = SpatialCNNExtractor,
            #        vf_features_extractor = HierarchicalCNN (задаётся через kwargs)
            super()._build(lr_schedule)
            # action_net = Linear(pi_features_dim, n_actions): нам это не нужно —
            # SpatialCNNExtractor уже выдаёт n_actions логитов.
            # Заменяем на Identity чтобы не добавлять случайно инициализированный
            # линейный слой, который испортит пространственные логиты.
            self.action_net = nn.Identity()

else:
    # Заглушка если sb3_contrib не установлен
    class SpatialMaskableActorCriticPolicy:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("sb3_contrib не установлен. Установи: pip install sb3-contrib")
