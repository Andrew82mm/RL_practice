"""
combine_datasets.py — объединяет несколько bc_dataset_partN.npz в один датасет.

Использование:
    python training/combine_datasets.py \\
        --parts models/bc_part1.npz models/bc_part2.npz models/bc_part3.npz \\
        --output models/bc_dataset.npz
"""

import argparse
import os
import sys

import numpy as np


def combine(input_paths: list[str], output_path: str) -> None:
    parts = []
    for p in input_paths:
        meta = np.load(p, allow_pickle=True)
        parts.append({
            "n_samples": int(meta["n_samples"]),
            "obs_shape": tuple(meta["obs_shape"]),
            "obs_path":  str(meta["obs_path"]),
            "act_path":  str(meta["act_path"]),
        })
        print(f"  Часть: {p}  →  {parts[-1]['n_samples']:,} пар")

    obs_shape = parts[0]["obs_shape"]
    for p in parts:
        assert tuple(p["obs_shape"]) == obs_shape, \
            f"Несовместимые obs_shape: {p['obs_shape']} vs {obs_shape}"

    total = sum(p["n_samples"] for p in parts)
    print(f"\nВсего пар: {total:,}  obs_shape: {obs_shape}")

    out_obs_path = output_path.replace(".npz", "_obs.dat")
    out_act_path = output_path.replace(".npz", "_act.dat")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    out_obs = np.memmap(out_obs_path, dtype=np.float32, mode="w+", shape=(total, *obs_shape))
    out_act = np.memmap(out_act_path, dtype=np.int64,   mode="w+", shape=(total,))

    offset = 0
    for p in parts:
        n = p["n_samples"]
        obs_mm = np.memmap(p["obs_path"], dtype=np.float32, mode="r", shape=(n, *obs_shape))
        act_mm = np.memmap(p["act_path"], dtype=np.int64,   mode="r", shape=(n,))

        chunk = 50_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            out_obs[offset + start: offset + end] = obs_mm[start:end]
            out_act[offset + start: offset + end] = act_mm[start:end]

        del obs_mm, act_mm
        offset += n
        print(f"  Скопировано {n:,} пар из {p['obs_path']}")

    out_obs.flush()
    out_act.flush()
    del out_obs, out_act

    np.savez(output_path,
             n_samples=np.array(total),
             obs_shape=np.array(obs_shape),
             obs_path=np.array(out_obs_path),
             act_path=np.array(out_act_path))

    size_gb = total * int(np.prod(obs_shape)) * 4 / 1024**3
    print(f"\nГотово: {output_path}")
    print(f"Размер obs на диске: ~{size_gb:.1f} ГБ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Объединение BC-датасетов из частей")
    parser.add_argument(
        "--parts", nargs="+", required=True,
        help="Пути к .npz файлам частей (в нужном порядке)"
    )
    parser.add_argument(
        "--output", default="./models/bc_dataset.npz",
        help="Путь для сохранения объединённого датасета (default: ./models/bc_dataset.npz)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    combine(args.parts, args.output)
