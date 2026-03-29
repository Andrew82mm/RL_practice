"""
baselines.py — базовые агенты для сравнения с обученной PPO-моделью.

Агенты:
    RandomAgent     — случайный выбор из валидных действий.
    HeuristicAgent  — жадный математический алгоритм на основе оценки поля.

Математическая основа HeuristicAgent:
    Подход вдохновлён алгоритмом Dellacherie (Tetris, 2003) и работами
    Thiery & Scherrer (2009) по линейной комбинации признаков доски.
    Адаптирован под block puzzle: очищаются и строки, И столбцы одновременно,
    поэтому признаки симметричны по осям.

    На каждом шаге для каждого валидного хода симулируется размещение фигуры
    и результирующее поле оценивается по взвешенной сумме 7 признаков:

        score = Σ wᵢ · fᵢ(board_after_placement)

    Признаки (fᵢ):
        1. lines_cleared   — кол-во очищенных линий (строк + столбцов)
        2. combo_bonus     — бонус за одновременную очистку ≥2 линий
        3. perfect_clear   — бонус за полное очищение доски
        4. holes           — «дыры» (BFS-flood-fill от границ; пустые клетки,
                             недостижимые снаружи — гарантированно не очистятся)
        5. near_complete   — строки/столбцы с заполненностью ≥ (n-1)/n
                             (один шаг до очистки — сигнал хорошей позиции)
        6. adjacency       — кол-во пар смежных заполненных клеток (компактность)
        7. fill_variance   — дисперсия заполнения по строкам и столбцам
                             (штраф за неравномерность → поощряет равномерное заполнение)

    Выбирается действие с максимальным score.

Использование:
    from baselines import RandomAgent, HeuristicAgent
    agent = HeuristicAgent()
    action = agent.select_action(env)
"""

from __future__ import annotations
import numpy as np
from block_puzzle_env.environment import BlockPuzzleEnv


# ================================================================== #
#  1. RandomAgent
# ================================================================== #

class RandomAgent:
    """
    Случайный агент.
    На каждом шаге выбирает равномерно случайное действие из множества
    валидных (маска действий гарантирует, что невалидные ходы не выбираются).
    Служит нижней границей производительности.
    """

    def select_action(self, env: BlockPuzzleEnv) -> int:
        mask = env.action_masks()
        valid = np.where(mask)[0]
        return int(np.random.choice(valid))


# ================================================================== #
#  2. HeuristicAgent
# ================================================================== #

class HeuristicAgent:
    """
    Жадный эвристический агент.
    Для каждого валидного хода симулирует размещение и оценивает
    получившееся поле по взвешенной комбинации признаков (см. docstring модуля).
    """

    # Веса признаков (подобраны эмпирически)
    DEFAULT_WEIGHTS = {
        "lines_cleared":  10.0,   # главный сигнал — очищать линии
        "combo_bonus":     5.0,   # дополнительный бонус за ≥2 линий за раз
        "perfect_clear":  50.0,   # очень редкое и ценное событие
        "holes":          -6.0,   # штраф за «дыры» — они никогда не очистятся
        "near_complete":   3.0,   # бонус за почти заполненные линии
        "adjacency":       0.4,   # компактность помогает заполнять линии
        "fill_variance":  -0.6,   # поощряем равномерное заполнение доски
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self.w = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)

    # -------------------------------------------------------------- #
    # Признаки доски
    # -------------------------------------------------------------- #

    @staticmethod
    def _simulate(
        grid: np.ndarray,
        piece: np.ndarray,
        x: int,
        y: int,
    ) -> tuple[np.ndarray, int, bool]:
        """
        Симулирует размещение фигуры piece по координатам (x, y) на копии grid.
        Повторяет логику Board.place_piece + Board.clear_lines_and_score.

        Returns:
            new_grid      — доска после размещения и очистки
            lines_cleared — кол-во очищенных строк + столбцов
            is_perfect    — True если доска полностью пуста после очистки
        """
        g = grid.copy()
        h, w = piece.shape
        g[y : y + h, x : x + w] += piece

        # Одновременная очистка строк и столбцов (как в environment.py)
        filled_rows = np.where(np.all(g == 1, axis=1))[0]
        filled_cols = np.where(np.all(g == 1, axis=0))[0]
        lines = len(filled_rows) + len(filled_cols)

        if len(filled_rows):
            g[filled_rows, :] = 0
        if len(filled_cols):
            g[:, filled_cols] = 0

        return g, lines, bool(np.sum(g) == 0)

    @staticmethod
    def _holes(grid: np.ndarray) -> int:
        """
        Подсчёт «дыр» — пустых клеток, недоступных с границы поля.

        Алгоритм: BFS-flood-fill от всех граничных пустых клеток.
        Незатронутые пустые клетки — потенциально «замурованы» и
        не могут участвовать в очистке линий.
        """
        s = grid.shape[0]
        vis = np.zeros((s, s), dtype=bool)
        queue: list[tuple[int, int]] = []

        # Инициализация: все пустые граничные клетки
        for i in range(s):
            for j in (0, s - 1):
                if grid[i, j] == 0 and not vis[i, j]:
                    vis[i, j] = True
                    queue.append((i, j))
                if grid[j, i] == 0 and not vis[j, i]:
                    vis[j, i] = True
                    queue.append((j, i))

        # BFS
        while queue:
            r, c = queue.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < s and 0 <= nc < s and not vis[nr, nc] and grid[nr, nc] == 0:
                    vis[nr, nc] = True
                    queue.append((nr, nc))

        # Дыры = пустые клетки, не достигнутые BFS
        return int(np.sum((grid == 0) & ~vis))

    @staticmethod
    def _adjacency(grid: np.ndarray) -> int:
        """Кол-во пар соседних (по горизонтали и вертикали) заполненных клеток."""
        h = int(np.sum(grid[:, :-1] & grid[:, 1:]))
        v = int(np.sum(grid[:-1, :] & grid[1:, :]))
        return h + v

    @staticmethod
    def _near_complete(grid: np.ndarray) -> int:
        """
        Кол-во строк и столбцов с заполненностью ≥ (n-1).
        Такие линии находятся в одном ходу от очистки — ценный сигнал позиции.
        """
        n = grid.shape[0]
        threshold = n - 1
        rows = int(np.sum(np.sum(grid, axis=1) >= threshold))
        cols = int(np.sum(np.sum(grid, axis=0) >= threshold))
        return rows + cols

    def _score(self, grid: np.ndarray, lines: int, perfect: bool) -> float:
        """Взвешенная оценка состояния поля после хода."""
        w = self.w
        s = 0.0

        # Немедленная очистка линий
        s += w["lines_cleared"] * lines
        if lines >= 2:
            s += w["combo_bonus"] * lines * (lines - 1) / 2

        # Perfect clear
        if perfect:
            s += w["perfect_clear"]

        # Штраф за дыры
        s += w["holes"] * self._holes(grid)

        # Почти заполненные линии
        s += w["near_complete"] * self._near_complete(grid)

        # Компактность
        s += w["adjacency"] * self._adjacency(grid)

        # Дисперсия заполнения строк и столбцов (оба направления симметричны)
        row_fills = np.sum(grid, axis=1).astype(float)
        col_fills = np.sum(grid, axis=0).astype(float)
        s += w["fill_variance"] * (float(np.var(row_fills)) + float(np.var(col_fills)))

        return s

    # -------------------------------------------------------------- #
    # Выбор действия
    # -------------------------------------------------------------- #

    def select_action(self, env: BlockPuzzleEnv) -> int:
        """
        Жадный выбор: перебираем все валидные ходы, симулируем каждый
        и выбираем с максимальной оценкой поля.

        Сложность: O(|valid_actions| * (n² + BFS(n²)))
        На поле 8×8 с ~192 действиями и BFS O(64) = практически мгновенно.
        """
        mask   = env.action_masks()
        valid  = np.where(mask)[0]
        size   = env.board_size
        pool   = env.piece_pool
        pieces = env.current_pieces
        grid   = env.board.grid

        best_score  = -np.inf
        best_action = int(valid[0])

        for action in valid:
            slot   = action // (size * size)
            coords = action % (size * size)
            y, x   = divmod(coords, size)

            new_grid, lines, perfect = self._simulate(grid, pool[pieces[slot]], x, y)
            score = self._score(new_grid, lines, perfect)

            if score > best_score:
                best_score  = score
                best_action = int(action)

        return best_action
