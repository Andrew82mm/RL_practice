"""
bc_train.py — Behavior Cloning: supervised pre-training политики от HeuristicAgent.

Проблема на 16×16:
    Случайный старт PPO на 16×16 даёт <1 line/эпизод против ~60 у эвристики.
    Агент миллионами шагов блуждает в "мёртвой зоне" не получая полезного сигнала.

Решение — двухфазное обучение:
    Фаза 1 (этот файл): BC
        - Собираем N пар (obs, action) от HeuristicAgent
        - Обучаем политику supervised: CE loss на distribution logits
        - Политика стартует сразу с ~50-60 lines/эпизод
    Фаза 2 (run_training.py --bc-pretrained):
        - Загружаем веса из фазы 1 в MaskablePPO
        - Fine-tune PPO от хорошей точки старта

Использование:
    # Фаза 1: собрать данные + обучить BC
    python training/bc_train.py

    # Только собрать данные (если уже есть датасет)
    python training/bc_train.py --only-collect

    # Только обучить (если данные уже собраны)
    python training/bc_train.py --only-train

    # Фаза 2: PPO с BC-инициализацией
    python training/run_training.py --arch cnn --bc-pretrained ./models/bc_pretrained.zip
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from block_puzzle_env.environment import BlockPuzzleEnv
from evaluation.baselines import HeuristicAgent
from utils.cnn_extractor import HierarchicalCNN
from config import BC_CONFIG, CNN_TRAIN, LOGGING


def _make_mask_fn(env) -> np.ndarray:
    return env.action_masks()


# ===========================================================================
# Фаза 1: сбор данных
# ===========================================================================

def collect_dataset(n_samples: int, save_path: str) -> None:
    """
    Прогоняет HeuristicAgent и собирает (obs, action) пары.
    Данные сохраняются как .npz для переиспользования.
    """
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    print(f"[bc_train] Сбор датасета: {n_samples:,} пар от HeuristicAgent...")
    env = BlockPuzzleEnv()
    agent = HeuristicAgent()

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []

    obs, info = env.reset()
    total = 0
    episodes = 0
    ep_lines: list[int] = []
    ep_lines_buf = 0

    bar = tqdm(total=n_samples, unit="шаг", ncols=80, desc="Сбор") if _tqdm_available else None

    while total < n_samples:
        action = agent.select_action(env)
        obs_list.append(obs.copy())
        act_list.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_lines_buf += info.get("lines_cleared", 0)
        total += 1

        if bar is not None:
            bar.update(1)

        if terminated or truncated:
            episodes += 1
            ep_lines.append(ep_lines_buf)
            ep_lines_buf = 0
            obs, info = env.reset()

            if bar is not None:
                avg = np.mean(ep_lines[-20:]) if ep_lines else 0.0
                bar.set_postfix(episodes=episodes, lines_avg=f"{avg:.1f}")
            elif episodes % 50 == 0:
                avg = np.mean(ep_lines[-50:]) if ep_lines else 0.0
                pct = 100 * total / n_samples
                print(f"  [{pct:5.1f}%] шагов: {total:,}/{n_samples:,}  "
                      f"эпизодов: {episodes}  lines_avg(50): {avg:.1f}")

    if bar is not None:
        bar.close()

    avg_all = np.mean(ep_lines) if ep_lines else 0.0
    print(f"[bc_train] Собрано: {total:,} шагов, {episodes} эпизодов, "
          f"среднее lines: {avg_all:.1f}")

    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)   # (N, C, H, W)
    act_arr = np.array(act_list, dtype=np.int64)              # (N,)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez_compressed(save_path, obs=obs_arr, actions=act_arr)
    print(f"[bc_train] Датасет сохранён: {save_path} ({obs_arr.shape})")


# ===========================================================================
# Фаза 2: supervised обучение политики
# ===========================================================================

def train_bc(
    dataset_path: str,
    pretrained_path: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> None:
    """
    Supervised обучение CNN-политики на BC-датасете.
    Сохраняет политику как MaskablePPO .zip (без оптимайзера — только веса).

    Стратегия:
      - Создаём MaskablePPO с HierarchicalCNN
      - Обучаем только policy network (actor), critic не трогаем
      - Loss: CrossEntropy(policy_logits, expert_action)
      - После N эпох сохраняем модель через model.save()
    """
    print(f"[bc_train] Загружаем датасет: {dataset_path}")
    data = np.load(dataset_path)
    obs_np  = data["obs"]       # (N, C, H, W)
    acts_np = data["actions"]   # (N,)
    N = len(acts_np)
    print(f"[bc_train] Датасет: {N:,} пар, obs shape {obs_np.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bc_train] Устройство: {device}")

    obs_t  = torch.from_numpy(obs_np).to(device)
    acts_t = torch.from_numpy(acts_np).to(device)

    dataset    = TensorDataset(obs_t, acts_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Создаём модель чтобы получить правильную архитектуру политики
    env = BlockPuzzleEnv()
    env = ActionMasker(env, _make_mask_fn)

    policy_kwargs = {
        "net_arch": CNN_TRAIN["net_arch"],
        "features_extractor_class":  HierarchicalCNN,
        "features_extractor_kwargs": {
            "features_dim": CNN_TRAIN.get("features_dim", 512),
        },
    }

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    policy = model.policy.to(device)
    policy.train()

    # Оптимизируем только параметры актор-головы + feature extractor (не critic)
    # Если хочется обучать и critic — убери фильтр ниже
    pi_params = (
        list(policy.pi_features_extractor.parameters()) +
        list(policy.mlp_extractor.policy_net.parameters()) +
        list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(pi_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"\n[bc_train] Обучаем {n_epochs} эпох, batch_size={batch_size}, lr={lr}")
    print(f"[bc_train] Параметры актора: {sum(p.numel() for p in pi_params):,}")

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for obs_batch, acts_batch in dataloader:
            optimizer.zero_grad()

            # Forward через actor-path политики
            features = policy.pi_features_extractor(obs_batch)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)          # (B, n_actions)

            loss = criterion(logits, acts_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(pi_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(dim=1) == acts_batch).sum().item()
            n_batches += 1

        scheduler.step()

        acc = epoch_correct / (n_batches * batch_size)
        avg_loss = epoch_loss / n_batches
        if acc > best_acc:
            best_acc = acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Эпоха {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}  acc={acc:.3f}  best_acc={best_acc:.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\n[bc_train] BC обучение завершено. Лучший acc = {best_acc:.3f}")

    # Переносим обученные веса обратно в model.policy (на CPU)
    policy.to("cpu")
    model.policy.load_state_dict(policy.state_dict())

    # Сохраняем через SB3 API — получаем .zip совместимый с MaskablePPO.load()
    os.makedirs(os.path.dirname(pretrained_path) or ".", exist_ok=True)
    model.save(pretrained_path)
    print(f"[bc_train] BC-модель сохранена: {pretrained_path}.zip")
    print(f"[bc_train] Запуск PPO: python training/run_training.py --arch cnn --bc-pretrained {pretrained_path}.zip")

    env.close()


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior Cloning: supervised pre-training от HeuristicAgent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--only-collect", action="store_true",
        help="Только собрать датасет (не обучать)"
    )
    parser.add_argument(
        "--only-train", action="store_true",
        help="Только обучить BC (датасет уже собран)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=BC_CONFIG["n_samples"],
        help=f"Число пар для сбора (default: {BC_CONFIG['n_samples']:,})"
    )
    parser.add_argument(
        "--dataset-path", type=str, default=BC_CONFIG["dataset_path"],
        help=f"Путь к датасету (default: {BC_CONFIG['dataset_path']})"
    )
    parser.add_argument(
        "--pretrained-path", type=str, default=BC_CONFIG["pretrained_path"],
        help=f"Путь для сохранения BC-модели (default: {BC_CONFIG['pretrained_path']})"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=BC_CONFIG["n_epochs"],
        help=f"Число эпох supervised обучения (default: {BC_CONFIG['n_epochs']})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BC_CONFIG["batch_size"],
        help=f"Batch size (default: {BC_CONFIG['batch_size']})"
    )
    parser.add_argument(
        "--lr", type=float, default=BC_CONFIG["lr"],
        help=f"Learning rate (default: {BC_CONFIG['lr']})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.only_train:
        collect_dataset(args.n_samples, args.dataset_path)

    if not args.only_collect:
        train_bc(
            dataset_path=args.dataset_path,
            pretrained_path=args.pretrained_path,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=BC_CONFIG["weight_decay"],
        )
