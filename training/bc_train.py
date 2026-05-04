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
from utils.vit_extractor import SmallViT
from config import BC_CONFIG, CNN_TRAIN, VIT_TRAIN, LOGGING


def _make_mask_fn(env) -> np.ndarray:
    return env.action_masks()


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, obs_mm, acts_mm):
        self.obs  = obs_mm
        self.acts = acts_mm
    def __len__(self):
        return len(self.acts)
    def __getitem__(self, idx):
        return torch.from_numpy(self.obs[idx].copy()), int(self.acts[idx])


# ===========================================================================
# Фаза 1: сбор данных
# ===========================================================================

def collect_dataset(n_samples: int, save_path: str) -> None:
    """
    Прогоняет HeuristicAgent и собирает (obs, action) пары.

    Пишет сразу на диск через np.memmap — не держит весь датасет в RAM.
    500k × (15,16,16) × float32 = 7.68 ГБ в RAM → OOM.
    С memmap: пиковое потребление RAM ~несколько МБ независимо от n_samples.
    """
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    from block_puzzle_env.environment import N_CHANNELS
    from config import ENV

    n = ENV["board_size"]
    obs_shape = (N_CHANNELS, n, n)
    obs_bytes  = n_samples * N_CHANNELS * n * n * 4  / 1024**3
    print(f"[bc_train] Сбор датасета: {n_samples:,} пар от HeuristicAgent...")
    print(f"[bc_train] Размер на диске: ~{obs_bytes:.1f} ГБ (obs float32) + actions")

    # Предвыделяем memmap-файлы на диске — данные пишутся туда напрямую
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    obs_path = save_path.replace(".npz", "_obs.dat")
    act_path = save_path.replace(".npz", "_act.dat")

    obs_mm  = np.memmap(obs_path,  dtype=np.float32, mode="w+", shape=(n_samples, *obs_shape))
    acts_mm = np.memmap(act_path,  dtype=np.int64,   mode="w+", shape=(n_samples,))

    env = BlockPuzzleEnv()
    agent = HeuristicAgent()

    obs, info = env.reset()
    total = 0
    episodes = 0
    ep_lines: list[int] = []
    ep_lines_buf = 0

    bar = tqdm(total=n_samples, unit="шаг", ncols=80, desc="Сбор") if _tqdm_available else None

    while total < n_samples:
        action = agent.select_action(env)

        # Пишем прямо в memmap — нулевая доп. RAM
        obs_mm[total]  = obs
        acts_mm[total] = action

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

    # Сбрасываем на диск и закрываем
    obs_mm.flush()
    acts_mm.flush()
    del obs_mm, acts_mm

    avg_all = np.mean(ep_lines) if ep_lines else 0.0
    print(f"[bc_train] Собрано: {total:,} шагов, {episodes} эпизодов, "
          f"среднее lines: {avg_all:.1f}")

    # Сохраняем метаданные в .npz (только shape и пути к .dat файлам — сами данные уже на диске)
    np.savez(save_path,
             n_samples=np.array(n_samples),
             obs_shape=np.array(obs_shape),
             obs_path=np.array(obs_path),
             act_path=np.array(act_path))
    print(f"[bc_train] Метаданные сохранены: {save_path}")
    print(f"[bc_train] Данные: {obs_path} + {act_path}")


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
    checkpoint_every: int = 5,
    arch: str = "cnn",
) -> None:
    """
    Supervised обучение политики на BC-датасете.
    Сохраняет политику как MaskablePPO .zip (без оптимайзера — только веса).
    Поддерживает arch="cnn" (HierarchicalCNN) и arch="vit" (SmallViT).
    """
    print(f"[bc_train] Загружаем датасет: {dataset_path}")
    meta = np.load(dataset_path, allow_pickle=True)
    n_samples  = int(meta["n_samples"])
    obs_shape  = tuple(meta["obs_shape"])
    obs_path   = str(meta["obs_path"])
    act_path   = str(meta["act_path"])

    # Открываем memmap только для чтения — данные не грузятся в RAM целиком
    obs_mm  = np.memmap(obs_path, dtype=np.float32, mode="r", shape=(n_samples, *obs_shape))
    acts_mm = np.memmap(act_path, dtype=np.int64,   mode="r", shape=(n_samples,))
    N = n_samples
    print(f"[bc_train] Датасет: {N:,} пар, obs shape {(N, *obs_shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bc_train] Устройство: {device}  arch={arch}")

    dataset    = MemmapDataset(obs_mm, acts_mm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    env = BlockPuzzleEnv()
    env = ActionMasker(env, _make_mask_fn)

    if arch == "vit":
        train_cfg = VIT_TRAIN
        policy_kwargs = {
            "net_arch": train_cfg["net_arch"],
            "features_extractor_class":  SmallViT,
            "features_extractor_kwargs": {
                "features_dim": train_cfg["features_dim"],
                "embed_dim":    train_cfg["embed_dim"],
                "n_heads":      train_cfg["n_heads"],
                "n_layers":     train_cfg["n_layers"],
                "ffn_dim":      train_cfg["ffn_dim"],
            },
        }
    else:  # cnn
        train_cfg = CNN_TRAIN
        policy_kwargs = {
            "net_arch": train_cfg["net_arch"],
            "features_extractor_class":  HierarchicalCNN,
            "features_extractor_kwargs": {
                "features_dim": train_cfg["features_dim"],
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
            obs_batch  = obs_batch.to(device)
            acts_batch = acts_batch.to(device)

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

        if epoch % checkpoint_every == 0:
            policy.to("cpu")
            model.policy.load_state_dict(policy.state_dict())
            ckpt_path = f"{pretrained_path}_ckpt_ep{epoch}"
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            model.save(ckpt_path)
            policy.to(device)
            print(f"  [ckpt] Сохранён: {ckpt_path}.zip")

    print(f"\n[bc_train] BC обучение завершено. Лучший acc = {best_acc:.3f}")

    # Переносим обученные веса обратно в model.policy (на CPU)
    policy.to("cpu")
    model.policy.load_state_dict(policy.state_dict())

    # Сохраняем через SB3 API — получаем .zip совместимый с MaskablePPO.load()
    os.makedirs(os.path.dirname(pretrained_path) or ".", exist_ok=True)
    model.save(pretrained_path)
    print(f"[bc_train] BC-модель сохранена: {pretrained_path}.zip")
    print(f"[bc_train] Запуск PPO: python training/run_training.py --arch {arch} --bc-pretrained {pretrained_path}.zip")

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
    parser.add_argument(
        "--arch", type=str, default="cnn", choices=["cnn", "vit"],
        help="Архитектура политики: cnn (HierarchicalCNN) или vit (SmallViT) (default: cnn)"
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
            arch=args.arch,
        )
