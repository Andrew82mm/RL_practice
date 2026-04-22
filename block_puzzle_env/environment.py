import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from scipy.signal import correlate2d, fftconvolve

from .logic import Board
from .pieces import PIECE_POOL
from config import REWARD, ENV


class BlockPuzzleEnv(gym.Env):
    """
    Block Puzzle окружение.

    Observation Space:
        Box(float32, shape=(14, 8, 8)) — 14 каналов:
            0:    игровое поле (binary)
            1-3:  текущие три фигуры (форма в левом верхнем углу)
            4:    заполненность строк — row_fill[i] broadcast по всей строке i
            5:    заполненность столбцов — col_fill[j] broadcast по всему столбцу j
            6-8:  placement heatmap для каждой фигуры:
                  1.0 в (y,x) если фигуру можно поставить с верхним левым углом в (x,y)
            9-11: survivability heatmap для каждой фигуры:
                  для каждой валидной позиции фигуры i — сколько суммарно
                  валидных ходов остаётся у ОСТАЛЬНЫХ фигур после этого размещения,
                  нормировано на [0, 1]. Даёт lookahead внутри раунда.
            12:   largest empty blob map — бинарная маска крупнейшей связной
                  компоненты пустых клеток. Агент видит "территорию".
            13:   dead zone mask — бинарная маска клеток в пустых компонентах,
                  куда не влезет ни одна фигура из пула. Гарантированный балласт.
        ВАЖНО: CnnPolicy в stable-baselines3 требует float32.
        Канал (H, W) интерпретируется как (C, H, W).

    Action Space:
        Discrete(3 * 8 * 8 = 192)
        action = slot_idx * 64 + y * 8 + x

    Action Mask:
        Метод action_masks() возвращает bool-вектор длиной 192.
        MaskablePPO использует его через ActionMasker, исключая невалидные действия из сэмплирования — агент никогда не выбирает запрещённое действие.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.board_size = ENV["board_size"]
        self.board = Board(self.board_size)
        self.piece_pool = PIECE_POOL
        self._max_steps = ENV["max_steps"]
        self._step_count = 0

        self.action_space = spaces.Discrete(
            ENV["pieces_per_round"] * self.board_size * self.board_size
        )

        # float32 обязателен для CnnPolicy
        # 14 каналов (Gen 3):
        #   0:    игровое поле (binary)
        #   1-3:  формы текущих фигур (в левом верхнем углу)
        #   4:    заполненность строк (row_fill broadcast)
        #   5:    заполненность столбцов (col_fill broadcast)
        #   6-8:  placement heatmap для каждой фигуры
        #   9-11: survivability heatmap для каждой фигуры (lookahead внутри раунда)
        #   12:   largest empty blob map
        #   13:   dead zone mask
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(14, self.board_size, self.board_size),
            dtype=np.float32,
        )

        self.current_pieces: list[int] = []
        self.render_mode = render_mode

        # Статистика эпизода (для TensorBoard callback)
        self._ep_lines_cleared = 0
        self._ep_perfect_clears = 0
        self._ep_pieces_placed = 0

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((14, self.board_size, self.board_size), dtype=np.float32)
        n = self.board_size

        # Канал 0: игровое поле
        obs[0] = self.board.grid.astype(np.float32)

        # Каналы 1-3: формы фигур (в левом верхнем углу)
        for i, piece_idx in enumerate(self.current_pieces):
            piece = self.piece_pool[piece_idx]
            h, w = piece.shape
            obs[i + 1, :h, :w] = piece.astype(np.float32)

        # Канал 4: заполненность каждой строки (broadcast по ширине)
        row_fills = self.board.grid.sum(axis=1).astype(np.float32) / n
        obs[4] = row_fills[:, np.newaxis]

        # Канал 5: заполненность каждого столбца (broadcast по высоте)
        col_fills = self.board.grid.sum(axis=0).astype(np.float32) / n
        obs[5] = col_fills[np.newaxis, :]

        # Каналы 6-8: placement heatmap (из action_mask — нет дублирования can_place)
        mask = self.board.compute_action_mask(self.piece_pool, self.current_pieces, n)
        for i in range(len(self.current_pieces)):
            slot_mask = mask[i * n * n : (i + 1) * n * n]
            obs[6 + i] = slot_mask.reshape(n, n).astype(np.float32)

        # Каналы 9-11: survivability heatmap
        # Для каждой валидной позиции фигуры i:
        #   cell(y,x) = суммарное число валидных ходов для ОСТАЛЬНЫХ фигур
        #               после размещения фигуры i в (x,y), нормировано на [0,1].
        # Даёт агенту lookahead: "если поставлю сюда — сколько ходов останется другим".
        #
        # Реализация через scipy.signal.correlate2d:
        #   correlate2d(temp, piece_j, mode='valid')[ry, rx]
        #       = sum(temp[ry:ry+ph, rx:rx+pw] * piece_j)
        #   > 0 означает overlap → нельзя поставить; == 0 → можно.
        #   mode='valid' возвращает только позиции где фигура полностью в пределах доски.
        #   Один C-вызов вместо (n-ph+1)*(n-pw+1) Python-итераций.
        num_pieces = len(self.current_pieces)
        if num_pieces > 1:
            board_f32 = self.board.grid.astype(np.float32)

            for i in range(num_pieces):
                placement_map = obs[6 + i]
                remaining = [j for j in range(num_pieces) if j != i]
                pieces_j = [
                    self.piece_pool[self.current_pieces[j]].astype(np.float32)
                    for j in remaining
                ]
                norm = float(len(remaining) * n * n)

                piece_i = self.piece_pool[self.current_pieces[i]].astype(np.float32)
                hi, wi = piece_i.shape
                surv = np.zeros((n, n), dtype=np.float32)

                for pj in pieces_j:
                    hj, wj = pj.shape

                    # A_j[ry,rx] = 1 если piece_j влезает в (ry,rx) на текущей доске
                    A_j = (correlate2d(board_f32, pj, mode='valid') == 0).astype(np.float32)

                    # cross[k,l] = количество перекрывающихся клеток когда piece_j смещена
                    # на (k-(hj-1), l-(wj-1)) относительно piece_i.
                    # Позиции вне этого диапазона — не пересекаются → конфликта нет (B=1).
                    cross = correlate2d(piece_i, pj, mode='full')  # shape (hi+hj-1, wi+wj-1)
                    B_ext = np.ones((2 * n - 1, 2 * n - 1), dtype=np.float32)
                    dy0, dx0 = n - hj, n - wj
                    B_ext[dy0:dy0 + hi + hj - 1, dx0:dx0 + wi + wj - 1] = (cross == 0).astype(np.float32)

                    # fftconvolve(A_j, B_ext)[y+(n-1), x+(n-1)]
                    # = Σ_{ry,rx} A_j[ry,rx] * B_ext[y-ry+(n-1), x-rx+(n-1)]
                    # = количество позиций piece_j, валидных после размещения piece_i в (y,x)
                    conv = fftconvolve(A_j, B_ext)
                    surv[:n - hi + 1, :n - wi + 1] += conv[n - 1:2 * n - hi, n - 1:2 * n - wi]

                obs[9 + i] = surv * placement_map / norm

        # Каналы 12-13: крупнейший blob и мёртвые зоны — одним BFS проходом
        obs[12], obs[13] = self._compute_blob_and_dead_zone()

        return obs

    def _compute_blob_and_dead_zone(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Единственный BFS по пустым клеткам, вычисляет оба канала за один проход:
          - blob_ch (12): маска крупнейшей связной компоненты пустых клеток
          - dead_ch (13): маска компонент, куда не влезет ни одна текущая фигура

        Оптимизация: valid_pos для каждой фигуры вычисляется ОДИН РАЗ до BFS
        (в старом коде — заново для каждой компоненты × каждой фигуры).
        """
        n = self.board_size
        empty = (self.board.grid == 0)
        visited = np.zeros((n, n), dtype=bool)

        grid_f = self.board.grid.astype(np.float32)
        current_piece_mats = [self.piece_pool[idx].astype(np.float32) for idx in self.current_pieces]

        # Позиции где каждая фигура влезает на текущую доску — не зависит от компоненты
        valid_pos_per_piece = [
            (correlate2d(grid_f, piece, mode='valid') == 0)
            for piece in current_piece_mats
        ]

        best_component: list[tuple[int, int]] = []
        dead_cells: list[tuple[int, int]] = []

        for sy in range(n):
            for sx in range(n):
                if not empty[sy, sx] or visited[sy, sx]:
                    continue
                component: list[tuple[int, int]] = []
                q: deque[tuple[int, int]] = deque()
                q.append((sy, sx))
                visited[sy, sx] = True
                while q:
                    cy, cx = q.popleft()
                    component.append((cy, cx))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < n and 0 <= nx < n and empty[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))

                if len(component) > len(best_component):
                    best_component = component

                if current_piece_mats:
                    comp_mask = np.zeros((n, n), dtype=np.float32)
                    for y, x in component:
                        comp_mask[y, x] = 1.0

                    is_dead = True
                    for piece, valid_pos in zip(current_piece_mats, valid_pos_per_piece):
                        touches = (correlate2d(comp_mask, piece, mode='valid') > 0)
                        if np.any(valid_pos & touches):
                            is_dead = False
                            break

                    if is_dead:
                        dead_cells.extend(component)

        blob_ch = np.zeros((n, n), dtype=np.float32)
        for y, x in best_component:
            blob_ch[y, x] = 1.0

        dead_ch = np.zeros((n, n), dtype=np.float32)
        for y, x in dead_cells:
            dead_ch[y, x] = 1.0

        return blob_ch, dead_ch

    def _pieces_matrices(self) -> list[np.ndarray]:
        return [self.piece_pool[i] for i in self.current_pieces]

    def _generate_valid_pieces(self) -> list[int] | None:
        """Генерирует набор из 3 фигур, для которых есть хотя бы один ход."""
        for _ in range(200):
            indices = list(
                np.random.choice(len(self.piece_pool), size=ENV["pieces_per_round"], replace=True)
            )
            pieces = [self.piece_pool[i] for i in indices]
            if self.board.has_valid_moves(pieces):
                return indices
        return None

    def _ep_info(self) -> dict:
        """Словарь с метриками эпизода для callback."""
        return {
            "ep_lines_cleared": self._ep_lines_cleared,
            "ep_perfect_clears": self._ep_perfect_clears,
            "ep_pieces_placed": self._ep_pieces_placed,
        }

    # ------------------------------------------------------------------
    # Action Mask
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Bool-вектор длиной action_space.n.
        True  — действие валидно (можно поставить фигуру в эту позицию).
        False — запрещено.

        MaskablePPO использует маску при сэмплировании, поэтому агент физически не может выбрать невалидное действие — нет штрафов за неверный ход, обучение быстрее и стабильнее.
        """
        return self.board.compute_action_mask(
            self.piece_pool, self.current_pieces, self.board_size
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        self._step_count = 0
        self._ep_lines_cleared = 0
        self._ep_perfect_clears = 0
        self._ep_pieces_placed = 0

        self.current_pieces = self._generate_valid_pieces()
        if self.current_pieces is None:
            raise RuntimeError("Не удалось сгенерировать валидные фигуры на пустом поле!")

        obs = self._get_obs()
        info = {
            "action_masks": self.action_masks(),
            **self._ep_info(),
        }
        return obs, info

    def step(self, action: int):
        self._step_count += 1

        slot_idx = action // (self.board_size * self.board_size)
        coords   = action %  (self.board_size * self.board_size)
        y = coords // self.board_size
        x = coords %  self.board_size

        reward     = 0.0
        terminated = False
        truncated  = False

        # --- Защита от невалидного хода (не должна срабатывать с маской) ---
        if slot_idx >= len(self.current_pieces) or \
                not self.board.can_place(self.piece_pool[self.current_pieces[slot_idx]], x, y):
            reward = REWARD["invalid_move"]
            obs = self._get_obs()
            return obs, reward, terminated, truncated, {
                "action_masks": self.action_masks(),
                **self._ep_info(),
            }

        # --- Размещаем фигуру ---
        piece = self.piece_pool[self.current_pieces[slot_idx]]
        self.board.place_piece(piece, x, y)
        reward += REWARD["place_piece"]
        self._ep_pieces_placed += 1

        # --- Потенциальная награда за клетки (cell_potential) ---
        # Для каждой клетки фигуры: чем заполненнее её строка/столбец ПОСЛЕ
        # размещения, тем выше награда. Максимум = 1.0 когда клетка завершает
        # строку (row_fill=1.0) — непосредственно перед очисткой.
        # Даёт критику плотный state-dependent сигнал о прогрессе к очистке.
        cell_reward = 0.0
        for pr in range(piece.shape[0]):
            for pc in range(piece.shape[1]):
                if piece[pr, pc]:
                    r, c = y + pr, x + pc
                    row_fill = float(np.sum(self.board.grid[r])) / self.board_size
                    col_fill = float(np.sum(self.board.grid[:, c])) / self.board_size
                    cell_reward += (row_fill + col_fill) / 2.0
        reward += REWARD["cell_potential"] * cell_reward

        # --- Одновременная очистка строк и столбцов ---
        lines_cleared, is_perfect_clear = self.board.clear_lines_and_score()

        if lines_cleared > 0:
            # Combo: чем больше линий за раз, тем выгоднее.
            # combo_multiplier=1.0 — линейная шкала (без бонуса).
            # combo_multiplier=0   — НЕВЕРНО: 2+ линий дают нулевую награду (0^1=0).
            combo = REWARD["combo_multiplier"] if REWARD["combo_multiplier"] > 0 else 1.0
            reward += REWARD["line_cleared"] * lines_cleared * (combo ** (lines_cleared - 1))
            self._ep_lines_cleared += lines_cleared

        if is_perfect_clear:
            reward += REWARD["perfect_clear"]
            self._ep_perfect_clears += 1

        # --- Убираем использованную фигуру ---
        self.current_pieces.pop(slot_idx)

        # --- Проверяем game over СРАЗУ после хода ---
        remaining = self._pieces_matrices()
        if remaining and not self.board.has_valid_moves(remaining):
            # Оставшиеся фигуры некуда поставить — конец
            terminated = True
            occupied = int(np.sum(self.board.grid))
            reward += REWARD["game_over"] + occupied * REWARD["game_over_per_cell"]
            obs = self._get_obs()
            return obs, reward, terminated, truncated, {
                "lines_cleared": lines_cleared,
                "is_perfect_clear": is_perfect_clear,
                "action_masks": np.zeros(self.action_space.n, dtype=bool),
                **self._ep_info(),
            }

        # --- Если все фигуры текущего раунда использованы — генерируем новые ---
        if len(self.current_pieces) == 0:
            new_pieces = self._generate_valid_pieces()
            if new_pieces is None:
                terminated = True
                occupied = int(np.sum(self.board.grid))
                reward += REWARD["game_over"] + occupied * REWARD["game_over_per_cell"]
            else:
                reward += REWARD["round_complete"]
                self.current_pieces = new_pieces

        # --- Truncation по лимиту шагов ---
        if not terminated and self._step_count >= self._max_steps:
            truncated = True

        obs = self._get_obs()
        done = terminated or truncated
        info = {
            "lines_cleared": lines_cleared,
            "is_perfect_clear": is_perfect_clear,
            "action_masks": (
                np.zeros(self.action_space.n, dtype=bool)
                if done
                else self.action_masks()
            ),
            **self._ep_info(),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            symbols = {0: "·", 1: "█"}
            print("=== ПОЛЕ ===")
            for row in self.board.grid:
                print(" ".join(symbols[int(c)] for c in row))
            print(f"Шаг: {self._step_count} | "
                  f"Линий: {self._ep_lines_cleared} | "
                  f"Perfect Clears: {self._ep_perfect_clears}")
            print("=== ТЕКУЩИЕ ФИГУРЫ ===")
            for i, idx in enumerate(self.current_pieces):
                print(f"  [{i}] pool_idx={idx}")
                for row in self.piece_pool[idx]:
                    print("   " + " ".join("█" if c else "·" for c in row))
            print()
