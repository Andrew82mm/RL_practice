"""
statistical_tests.py — статистическое сравнение агентов.

Тесты:
    1. Mann–Whitney U      — непараметрический тест на доминирование распределений
    2. Cohen's d           — размер эффекта (effect size)
    3. Bootstrap CI        — доверительный интервал разницы средних
    4. Paired test         — агенты на одинаковых последовательностях фигур
    5. Survival analysis   — как далеко заходят агенты (сколько раундов выживают)

Запуск:
    python statistical_tests.py --model ./models/mlp_3_final.zip --n 500 --seed 42
    python statistical_tests.py --models ./models/mlp_3_final.zip ./models/cnn_2_final.zip --n 500
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import stats
from tqdm import tqdm

from block_puzzle_env.environment import BlockPuzzleEnv
from baselines import HeuristicAgent, RandomAgent


# ================================================================== #
#  Сбор сырых данных (не агрегированных)
# ================================================================== #

def collect_episodes(agent, n: int, seed: int) -> dict[str, np.ndarray]:
    """
    Прогоняет агента на n эпизодах, возвращает сырые массивы метрик.
    seed+ep гарантирует воспроизводимость и разные начальные состояния.
    """
    env = BlockPuzzleEnv()
    rewards, lines, pieces, steps = [], [], [], []

    for ep in tqdm(range(n), desc=f"  {_agent_name(agent):<20}", leave=False, ncols=72):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(env)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            done = terminated or truncated

        rewards.append(ep_reward)
        lines.append(info["ep_lines_cleared"])
        pieces.append(info["ep_pieces_placed"])
        steps.append(env._step_count)

    env.close()
    return {
        "reward": np.array(rewards),
        "lines":  np.array(lines),
        "pieces": np.array(pieces),
        "steps":  np.array(steps),
    }


def collect_paired(agent_a, agent_b, n: int, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Прогоняет оба агента на ОДИНАКОВЫХ эпизодах: одинаковый seed → одинаковые
    начальные состояния и последовательности фигур. Убирает variance от "везения".
    Возвращает пары (результат_A, результат_B) для каждого эпизода.
    """
    results_a: dict[str, list] = {"reward": [], "lines": [], "pieces": [], "steps": []}
    results_b: dict[str, list] = {"reward": [], "lines": [], "pieces": [], "steps": []}

    for ep in tqdm(range(n), desc="  Paired episodes", leave=False, ncols=72):
        ep_seed = seed + ep
        for agent, res in [(agent_a, results_a), (agent_b, results_b)]:
            env = BlockPuzzleEnv()
            obs, _ = env.reset(seed=ep_seed)
            done = False
            ep_reward = 0.0
            while not done:
                action = agent.select_action(env)
                obs, r, terminated, truncated, info = env.step(action)
                ep_reward += r
                done = terminated or truncated

            res["reward"].append(ep_reward)
            res["lines"].append(info["ep_lines_cleared"])
            res["pieces"].append(info["ep_pieces_placed"])
            res["steps"].append(env._step_count)
            env.close()

    return {
        metric: (np.array(results_a[metric]), np.array(results_b[metric]))
        for metric in ["reward", "lines", "pieces", "steps"]
    }


# ================================================================== #
#  Статистические тесты
# ================================================================== #

def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Mann–Whitney U test (Wilcoxon rank-sum).
    Непараметрический: не требует нормальности распределения.
    H0: P(A > B) = P(B > A) — распределения одинаковы.
    alternative='greater': A стохастически доминирует над B.
    """
    stat, p_two = stats.mannwhitneyu(a, b, alternative="two-sided")
    _, p_greater = stats.mannwhitneyu(a, b, alternative="greater")

    # Common Language Effect Size (CLES):
    # вероятность того что случайный A > случайного B
    cles = stat / (len(a) * len(b))

    return {
        "U":         stat,
        "p_twosided": p_two,
        "p_greater":  p_greater,
        "cles":       cles,         # 0.5 = нет эффекта, 1.0 = A всегда > B
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d — стандартизированный размер эффекта.
    d = (mean_A - mean_B) / pooled_std
    Интерпретация: 0.2=мало, 0.5=средне, 0.8=много, >1.2=очень много.
    """
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Bootstrap доверительный интервал для разницы средних (mean_A - mean_B).
    Не делает предположений о форме распределения.
    Возвращает (observed_diff, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    observed = np.mean(a) - np.mean(b)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.mean(sample_a) - np.mean(sample_b)

    alpha = 1 - ci
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return float(observed), lo, hi


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test для парных наблюдений.
    Используется когда оба агента играли на одинаковых эпизодах.
    Более мощный чем Mann–Whitney для парных данных.
    H0: медиана разностей (A - B) = 0.
    """
    diff = a - b
    if np.all(diff == 0):
        return {"stat": 0.0, "p": 1.0, "median_diff": 0.0, "win_rate": 0.5}

    stat, p = stats.wilcoxon(diff, alternative="two-sided")
    win_rate = float(np.mean(a > b))   # % эпизодов где A > B

    return {
        "stat":        stat,
        "p":           p,
        "median_diff": float(np.median(diff)),
        "win_rate":    win_rate,   # 0.5 = паритет, 1.0 = A всегда побеждает
    }


def survival_analysis(data: dict[str, np.ndarray]) -> dict:
    """
    Анализ "выживаемости" — как далеко заходит агент.
    Считаем долю эпизодов достигших каждого порогового числа линий.
    Аналог Kaplan-Meier curve, но для игровых результатов.
    """
    lines = data["lines"]
    thresholds = [1, 5, 10, 15, 20, 25, 30]
    survival = {t: float(np.mean(lines >= t)) for t in thresholds}
    return {
        "survival_rates": survival,
        "median_lines":   float(np.median(lines)),
        "q25_lines":      float(np.percentile(lines, 25)),
        "q75_lines":      float(np.percentile(lines, 75)),
        "max_lines":      float(np.max(lines)),
    }


# ================================================================== #
#  Печать результатов
# ================================================================== #

def _agent_name(agent) -> str:
    if hasattr(agent, "_name"):
        return agent._name
    return agent.__class__.__name__


def _significance(p: float) -> str:
    if p < 0.001:
        return "*** (p<0.001)"
    if p < 0.01:
        return "**  (p<0.01)"
    if p < 0.05:
        return "*   (p<0.05)"
    return "ns  (p≥0.05)"


def _effect_label(d: float) -> str:
    d = abs(d)
    if d >= 1.2:
        return "очень большой"
    if d >= 0.8:
        return "большой"
    if d >= 0.5:
        return "средний"
    if d >= 0.2:
        return "малый"
    return "пренебрежимый"


def print_pairwise(name_a: str, name_b: str, data_a: dict, data_b: dict):
    """Полный отчёт по паре агентов (независимые эпизоды)."""
    sep = "─" * 64

    print(f"\n  {sep}")
    print(f"  {name_a}  vs  {name_b}")
    print(f"  {sep}")

    for metric in ["lines", "reward", "pieces"]:
        a, b = data_a[metric], data_b[metric]
        mw   = mann_whitney(a, b)
        d    = cohens_d(a, b)
        obs, lo, hi = bootstrap_ci(a, b)

        label = {"lines": "Lines cleared", "reward": "Reward", "pieces": "Pieces placed"}[metric]
        print(f"\n  [{label}]")
        print(f"    {name_a:<20}  mean={np.mean(a):6.2f}  std={np.std(a):5.2f}  median={np.median(a):5.1f}")
        print(f"    {name_b:<20}  mean={np.mean(b):6.2f}  std={np.std(b):5.2f}  median={np.median(b):5.1f}")
        print(f"    Mann–Whitney U:    {_significance(mw['p_twosided'])}  (p={mw['p_twosided']:.4f})")
        print(f"    CLES:              {mw['cles']:.3f}  (P({name_a}>{name_b})  | 0.5=паритет, 1.0=всегда лучше)")
        print(f"    Cohen's d:         {d:+.3f}  — эффект {_effect_label(d)}")
        print(f"    Bootstrap 95% CI:  diff={obs:+.2f}  [{lo:+.2f} .. {hi:+.2f}]", end="")
        print("  ✓ значим" if lo > 0 or hi < 0 else "  ✗ включает 0")


def print_paired(name_a: str, name_b: str, paired: dict):
    """Отчёт по парным тестам (одинаковые эпизоды)."""
    sep = "─" * 64
    print(f"\n\n  {sep}")
    print(f"  PAIRED TEST: {name_a}  vs  {name_b}  (одинаковые эпизоды)")
    print(f"  {sep}")
    print(f"  Смысл: seed зафиксирован → одинаковые начальные фигуры.")
    print(f"  Убираем дисперсию от 'везения' — остаётся только качество агента.\n")

    for metric in ["lines", "reward", "pieces"]:
        a, b = paired[metric]
        w    = wilcoxon_paired(a, b)
        diff = a - b

        label = {"lines": "Lines cleared", "reward": "Reward", "pieces": "Pieces placed"}[metric]
        print(f"  [{label}]")
        print(f"    Wilcoxon signed-rank:  {_significance(w['p'])}  (p={w['p']:.4f})")
        print(f"    Медиана разности:      {w['median_diff']:+.2f}  (положительное = {name_a} лучше)")
        print(f"    Win rate {name_a}:        {w['win_rate']:.1%}  эпизодов где {name_a} > {name_b}")
        print(f"    Среднее A-B:           {np.mean(diff):+.2f}")
        print()


def print_survival(name: str, surv: dict):
    """Таблица выживаемости агента."""
    print(f"\n  Survival: {name}")
    print(f"    Медиана lines: {surv['median_lines']:.1f}  "
          f"(Q25={surv['q25_lines']:.1f}, Q75={surv['q75_lines']:.1f}, max={surv['max_lines']:.0f})")
    rates = surv["survival_rates"]
    reached = "  ".join(f"≥{t}: {rates[t]:.0%}" for t in sorted(rates))
    print(f"    {reached}")


def print_survival_comparison(names: list[str], survs: list[dict]):
    """Сравнительная таблица выживаемости."""
    sep = "─" * 72
    thresholds = sorted(next(iter(survs))["survival_rates"])

    print(f"\n\n  {sep}")
    print(f"  SURVIVAL TABLE — доля эпизодов достигших N линий")
    print(f"  {sep}")
    header = f"  {'Агент':<22}" + "".join(f"  ≥{t:>2}" for t in thresholds)
    print(header)
    print(f"  {'-'*68}")
    for name, surv in zip(names, survs):
        row = f"  {name:<22}" + "".join(
            f"  {surv['survival_rates'][t]:>4.0%}" for t in thresholds
        )
        print(row)
    print(f"  {sep}")


# ================================================================== #
#  CLI
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Block Puzzle RL — статистическое сравнение агентов"
    )
    parser.add_argument("--model",  type=str, default=None, help="Путь к .zip модели")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Несколько моделей")
    parser.add_argument("--n",      type=int, default=500,  help="Число эпизодов (default: 500)")
    parser.add_argument("--seed",   type=int, default=42,   help="Базовый seed (default: 42)")
    parser.add_argument("--no-paired", action="store_true", help="Пропустить парные тесты (быстрее)")
    parser.add_argument("--no-random", action="store_true", help="Не включать RandomAgent")
    args = parser.parse_args()

    # ── Агенты ───────────────────────────────────────────────────────
    agents: dict[str, object] = {}
    if not args.no_random:
        agents["Random"] = RandomAgent()
    agents["Heuristic"] = HeuristicAgent()

    model_paths = []
    if args.model:
        model_paths.append(args.model)
    if args.models:
        model_paths.extend(args.models)

    for path in model_paths:
        from evaluate import PPOAgent
        name = os.path.splitext(os.path.basename(path))[0]
        agents[name] = PPOAgent(path, name=name)

    if len(agents) < 2:
        print("Нужно хотя бы 2 агента для сравнения. Добавь --model путь/к/модели.zip")
        return

    # ── Сбор независимых данных ──────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  STATISTICAL TESTS")
    print(f"  Episodes: {args.n}  |  Seed: {args.seed}")
    print(f"{'='*64}\n")

    print("  [1/3] Сбор независимых эпизодов...")
    all_data: dict[str, dict] = {}
    for name, agent in agents.items():
        all_data[name] = collect_episodes(agent, n=args.n, seed=args.seed)
        surv = survival_analysis(all_data[name])
        print(f"  {name}: mean_lines={np.mean(all_data[name]['lines']):.2f}  "
              f"median={surv['median_lines']:.1f}  max={surv['max_lines']:.0f}")

    # ── Попарные тесты (независимые эпизоды) ────────────────────────
    print(f"\n  [2/3] Попарные тесты (независимые эпизоды)...")
    names = list(agents.keys())
    # Главное сравнение: каждая модель vs Heuristic
    baseline = "Heuristic"
    for name in names:
        if name == baseline:
            continue
        print_pairwise(name, baseline, all_data[name], all_data[baseline])

    # ── Парные тесты (одинаковые эпизоды) ───────────────────────────
    if not args.no_paired:
        print(f"\n\n  [3/3] Парные тесты (одинаковые начальные условия)...")
        for name in names:
            if name == baseline:
                continue
            paired = collect_paired(agents[name], agents[baseline], n=args.n, seed=args.seed)
            print_paired(name, baseline, paired)

    # ── Survival table ───────────────────────────────────────────────
    survs = [survival_analysis(all_data[n]) for n in names]
    print_survival_comparison(names, survs)

    print(f"\n  {'─'*64}")
    print(f"  Интерпретация:")
    print(f"    p < 0.05  → разница статистически значима")
    print(f"    CLES > 0.6 → умеренное доминирование; > 0.7 → сильное")
    print(f"    Cohen's d: 0.2=мало, 0.5=средне, 0.8=много")
    print(f"    Bootstrap CI не включает 0 → разница надёжная")
    print(f"    Win rate > 60% в paired test → агент стабильно лучше на любых фигурах")
    print(f"  {'─'*64}\n")


if __name__ == "__main__":
    main()
