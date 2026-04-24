import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from scipy.signal import correlate2d
from scipy.fft import next_fast_len, rfft2, irfft2

from .logic import Board
from .pieces import PIECE_POOL
from config import REWARD, ENV

try:
    from block_puzzle_env._fast import compute_blob_and_dead_zone as _fast_bnd
    _FAST_BFS = True
except ImportError:
    _FAST_BFS = False


# Число каналов наблюдения (Gen 4):
#   0:    игровое поле (binary)
#   1-3:  текущие три фигуры (форма в левом верхнем углу)
#   4:    заполненность строк — row_fill[i] broadcast по всей строке i
#   5:    заполненность столбцов — col_fill[j] broadcast по всему столбцу j
#   6-8:  placement heatmap для каждой фигуры
#   9-11: survivability heatmap для каждой фигуры (lookahead внутри раунда)
#   12:   largest empty blob map
#   13:   dead zone mask
#   14:   holes-from-border mask (пустые клетки, недостижимые с границы поля)
N_CHANNELS = 15


class BlockPuzzleEnv(gym.Env):
    """
    Block Puzzle окружение.

    Observation Space:
        Box(float32, shape=(15, board_size, board_size))

    Action Space:
        Discrete(3 * board_size * board_size)
        action = slot_idx * board_size² + y * board_size + x
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

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(N_CHANNELS, self.board_size, self.board_size),
            dtype=np.float32,
        )

        self.current_pieces: list[int] = []
        self.render_mode = render_mode

        self._ep_lines_cleared = 0
        self._ep_perfect_clears = 0
        self._ep_pieces_placed = 0

        # Кэш FFT(B_ext) для survivability: ключ (i_pool_idx, j_pool_idx).
        self._surv_fft_cache: dict = {}

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        n = self.board_size
        obs = np.zeros((N_CHANNELS, n, n), dtype=np.float32)

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

        # Каналы 6-8: placement heatmap
        mask = self.board.compute_action_mask(self.piece_pool, self.current_pieces, n)
        for i in range(len(self.current_pieces)):
            slot_mask = mask[i * n * n : (i + 1) * n * n]
            obs[6 + i] = slot_mask.reshape(n, n).astype(np.float32)

        # Каналы 9-11: survivability heatmap + каналы 12-13: blob / dead-zone
        num_pieces = len(self.current_pieces)
        board_f32 = self.board.grid.astype(np.float32)
        current_piece_mats = [self.piece_pool[idx].astype(np.float32) for idx in self.current_pieces]

        # A_list[k]: где фигура k влезает на текущую доску
        A_list = [
            (correlate2d(board_f32, piece, mode='valid') == 0).astype(np.float32)
            for piece in current_piece_mats
        ]

        if num_pieces > 1:
            for i in range(num_pieces):
                placement_map = obs[6 + i]
                piece_i = current_piece_mats[i]
                hi, wi = piece_i.shape
                i_pool = self.current_pieces[i]
                norm = float((num_pieces - 1) * n * n)
                surv = np.zeros((n, n), dtype=np.float32)

                for j in range(num_pieces):
                    if j == i:
                        continue
                    j_pool = self.current_pieces[j]
                    pj = current_piece_mats[j]
                    hj, wj = pj.shape
                    A_j = A_list[j]

                    cache_key = (i_pool, j_pool)
                    if cache_key not in self._surv_fft_cache:
                        cross = correlate2d(piece_i, pj, mode='full')
                        B_ext = np.ones((2 * n - 1, 2 * n - 1), dtype=np.float32)
                        dy0, dx0 = n - hj, n - wj
                        B_ext[dy0:dy0+hi+hj-1, dx0:dx0+wi+wj-1] = (cross == 0).astype(np.float32)
                        out_h = A_j.shape[0] + B_ext.shape[0] - 1
                        out_w = A_j.shape[1] + B_ext.shape[1] - 1
                        fft_shape = (next_fast_len(out_h), next_fast_len(out_w))
                        self._surv_fft_cache[cache_key] = (rfft2(B_ext, fft_shape), fft_shape, out_h, out_w)

                    fft_B, fft_shape, out_h, out_w = self._surv_fft_cache[cache_key]
                    conv = irfft2(rfft2(A_j, fft_shape) * fft_B, fft_shape)[:out_h, :out_w]
                    surv[:n - hi + 1, :n - wi + 1] += conv[n - 1:2 * n - hi, n - 1:2 * n - wi]

                obs[9 + i] = surv * placement_map / norm

        # Каналы 12-13: blob и dead-zone
        obs[12], obs[13] = self._compute_blob_and_dead_zone(A_list, current_piece_mats)

        # Канал 14: holes-from-border (пустые клетки, недостижимые с границы)
        obs[14] = self._compute_holes_from_border()

        return obs

    def _compute_blob_and_dead_zone(
        self,
        A_list: list,
        current_piece_mats: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Единственный BFS по пустым клеткам, вычисляет blob и dead-zone."""
        valid_pos_list = [a.astype(np.uint8) for a in A_list]
        piece_offset_arrays = [
            np.argwhere(piece > 0).astype(np.int32)
            for piece in current_piece_mats
        ]

        if _FAST_BFS:
            return _fast_bnd(self.board.grid, valid_pos_list, piece_offset_arrays)

        n = self.board_size
        empty = (self.board.grid == 0)
        visited = np.zeros((n, n), dtype=bool)
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
                    is_dead = True
                    for vp, off_arr in zip(valid_pos_list, piece_offset_arrays):
                        vp_h, vp_w = vp.shape
                        for cy, cx in component:
                            if not is_dead:
                                break
                            for pr, pc in off_arr:
                                py, px = cy - pr, cx - pc
                                if 0 <= py < vp_h and 0 <= px < vp_w and vp[py, px]:
                                    is_dead = False
                                    break
                        if not is_dead:
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

    def _compute_holes_from_border(self) -> np.ndarray:
        """
        BFS от всех граничных пустых клеток.
        Незатронутые пустые клетки — «дыры»: замурованы и никогда не очистятся.
        Это симметричный аналог признака holes в HeuristicAgent с весом −6.
        """
        n = self.board_size
        grid = self.board.grid
        visited = np.zeros((n, n), dtype=bool)
        queue: list[tuple[int, int]] = []

        for i in range(n):
            for j in (0, n - 1):
                if grid[i, j] == 0 and not visited[i, j]:
                    visited[i, j] = True
                    queue.append((i, j))
                if grid[j, i] == 0 and not visited[j, i]:
                    visited[j, i] = True
                    queue.append((j, i))

        while queue:
            r, c = queue.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and not visited[nr, nc] and grid[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        holes = ((grid == 0) & ~visited).astype(np.float32)
        return holes

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
        return {
            "ep_lines_cleared": self._ep_lines_cleared,
            "ep_perfect_clears": self._ep_perfect_clears,
            "ep_pieces_placed": self._ep_pieces_placed,
        }

    # ------------------------------------------------------------------
    # Action Mask
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
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
        n = self.board_size

        slot_idx = action // (n * n)
        coords   = action %  (n * n)
        y = coords // n
        x = coords %  n

        reward     = 0.0
        terminated = False
        truncated  = False

        if slot_idx >= len(self.current_pieces) or \
                not self.board.can_place(self.piece_pool[self.current_pieces[slot_idx]], x, y):
            reward = REWARD["invalid_move"]
            obs = self._get_obs()
            return obs, reward, terminated, truncated, {
                "action_masks": self.action_masks(),
                **self._ep_info(),
            }

        piece = self.piece_pool[self.current_pieces[slot_idx]]
        self.board.place_piece(piece, x, y)
        reward += REWARD["place_piece"]
        self._ep_pieces_placed += 1

        # cell_potential: для каждой клетки фигуры — насколько заполнены её строка/столбец
        cell_reward = 0.0
        for pr in range(piece.shape[0]):
            for pc in range(piece.shape[1]):
                if piece[pr, pc]:
                    r, c = y + pr, x + pc
                    row_fill = float(np.sum(self.board.grid[r])) / n
                    col_fill = float(np.sum(self.board.grid[:, c])) / n
                    cell_reward += (row_fill + col_fill) / 2.0
        reward += REWARD["cell_potential"] * cell_reward

        lines_cleared, is_perfect_clear = self.board.clear_lines_and_score()

        if lines_cleared > 0:
            combo = REWARD["combo_multiplier"] if REWARD["combo_multiplier"] > 0 else 1.0
            reward += REWARD["line_cleared"] * lines_cleared * (combo ** (lines_cleared - 1))
            self._ep_lines_cleared += lines_cleared

        if is_perfect_clear:
            reward += REWARD["perfect_clear"]
            self._ep_perfect_clears += 1

        self.current_pieces.pop(slot_idx)

        remaining = self._pieces_matrices()
        if remaining and not self.board.has_valid_moves(remaining):
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

        if len(self.current_pieces) == 0:
            new_pieces = self._generate_valid_pieces()
            if new_pieces is None:
                terminated = True
                occupied = int(np.sum(self.board.grid))
                reward += REWARD["game_over"] + occupied * REWARD["game_over_per_cell"]
            else:
                reward += REWARD["round_complete"]
                self.current_pieces = new_pieces

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
