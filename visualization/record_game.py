"""
record_game.py — записывает игру агента в JSON для визуализатора.

Использование:
    # PPO-модель
    python visualization/record_game.py \
        --model models/gen4/cnn_bc_finetune_final.zip \
        --agent ppo --n-episodes 3 --output visualization/replay_cnn.json

    # ViT
    python visualization/record_game.py \
        --model models/gen4/vit_bc_finetune_final.zip \
        --agent ppo --n-episodes 3 --output visualization/replay_vit.json

    # Эвристика
    python visualization/record_game.py \
        --agent heuristic --n-episodes 3 --output visualization/replay_heuristic.json

    # Рандом
    python visualization/record_game.py \
        --agent random --n-episodes 3 --output visualization/replay_random.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from block_puzzle_env.environment import BlockPuzzleEnv


def _load_ppo(model_path: str):
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(model_path)


def _ppo_action_and_logits(model, env: BlockPuzzleEnv):
    """Возвращает (action, logits_per_slot: list[n×n])."""
    n = env.board_size
    obs = env._get_obs()[np.newaxis]
    expected_ch = model.observation_space.shape[0]
    if obs.shape[1] != expected_ch:
        obs = obs[:, :expected_ch, :, :]

    mask   = env.action_masks()
    policy = model.policy

    with torch.no_grad():
        obs_t    = torch.as_tensor(obs, dtype=torch.float32)
        features = policy.extract_features(obs_t, policy.pi_features_extractor)
        latent   = policy.mlp_extractor.forward_actor(features)
        logits   = policy.action_net(latent)[0].cpu().numpy()

    masked = logits.copy()
    masked[~mask] = -1e9
    action = int(np.argmax(masked))

    logits_per_slot = []
    for s in range(3):
        slot_raw = logits[s * n * n: (s + 1) * n * n].reshape(n, n)
        logits_per_slot.append([[round(float(v), 4) for v in row] for row in slot_raw])

    return action, logits_per_slot


def _null_logits(n: int) -> list:
    """Пустые логиты для агентов без нейросети."""
    return [[[0.0] * n for _ in range(n)] for _ in range(3)]


def _record_episode(agent_fn, env: BlockPuzzleEnv, seed: int) -> list[dict]:
    """Прогоняет один эпизод, возвращает список фреймов."""
    n = env.board_size
    env.reset(seed=seed)
    frames: list[dict] = []
    done = False

    while not done:
        board_before = env.board.grid.tolist()
        pieces = [env.piece_pool[i].tolist() for i in env.current_pieces]

        mask_flat = env.action_masks()
        valid_masks = []
        for s in range(3):
            slot_mask = mask_flat[s * n * n: (s + 1) * n * n].reshape(n, n)
            valid_masks.append([[bool(v) for v in row] for row in slot_mask])

        action, logits_per_slot = agent_fn(env)

        slot_idx = action // (n * n)
        coords   = action % (n * n)
        ay       = coords // n
        ax       = coords % n

        _, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frames.append({
            "step":             len(frames),
            "board":            board_before,
            "board_after":      env.board.grid.tolist(),
            "pieces":           pieces,
            "action_slot":      slot_idx,
            "action_y":         ay,
            "action_x":         ax,
            "valid_masks":      valid_masks,
            "logits":           logits_per_slot,
            "reward":           round(float(reward), 3),
            "lines_cleared":    int(info.get("lines_cleared", 0)),
            "ep_lines_cleared": int(info["ep_lines_cleared"]),
            "ep_pieces_placed": int(info["ep_pieces_placed"]),
            "terminated":       bool(terminated),
            "truncated":        bool(truncated),
        })

    return frames


def record_episodes(
    agent_type: str,
    model_path: str | None,
    n_episodes: int,
    seed: int,
    output_path: str,
) -> None:
    env = BlockPuzzleEnv()
    n   = env.board_size

    # Фабрика функции агента
    if agent_type == "ppo":
        assert model_path, "--model обязателен для --agent ppo"
        print(f"[record] Загружаем PPO модель: {model_path}")
        model = _load_ppo(model_path)
        agent_name = os.path.basename(model_path).replace(".zip", "")
        def agent_fn(e):
            return _ppo_action_and_logits(model, e)

    elif agent_type == "heuristic":
        from evaluation.baselines import HeuristicAgent
        heuristic = HeuristicAgent()
        agent_name = "heuristic"
        def agent_fn(e):
            return heuristic.select_action(e), _null_logits(n)

    elif agent_type == "random":
        from evaluation.baselines import RandomAgent
        random_ag = RandomAgent()
        agent_name = "random"
        def agent_fn(e):
            return random_ag.select_action(e), _null_logits(n)

    else:
        raise ValueError(f"Неизвестный --agent: {agent_type}")

    print(f"[record] Агент: {agent_name}  |  эпизодов: {n_episodes}")

    all_episodes = []
    for ep in range(n_episodes):
        ep_seed = seed + ep
        print(f"  Эпизод {ep + 1}/{n_episodes} (seed={ep_seed})...", end="", flush=True)
        frames = _record_episode(agent_fn, env, ep_seed)
        ep_lines  = frames[-1]["ep_lines_cleared"] if frames else 0
        ep_pieces = frames[-1]["ep_pieces_placed"]  if frames else 0
        print(f" {len(frames)} ходов, {ep_lines} линий")
        all_episodes.append({
            "episode":          ep + 1,
            "seed":             ep_seed,
            "total_steps":      len(frames),
            "ep_lines_cleared": ep_lines,
            "ep_pieces_placed": ep_pieces,
            "frames":           frames,
        })

    env.close()

    data = {
        "model_name": agent_name,
        "agent_type": agent_type,
        "board_size":  n,
        "n_episodes":  n_episodes,
        "episodes":    all_episodes,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[record] → {output_path} ({size_mb:.1f} MB)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Записывает игру агента в JSON для вьюера")
    parser.add_argument("--agent", type=str, default="ppo",
                        choices=["ppo", "heuristic", "random"],
                        help="Тип агента (default: ppo)")
    parser.add_argument("--model", type=str, default=None,
                        help="Путь к .zip модели (только для --agent ppo)")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Число записываемых эпизодов (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Начальный seed (default: 42)")
    parser.add_argument("--output", type=str, default="visualization/replay.json",
                        help="Путь к выходному JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    record_episodes(
        agent_type=args.agent,
        model_path=args.model,
        n_episodes=args.n_episodes,
        seed=args.seed,
        output_path=args.output,
    )
