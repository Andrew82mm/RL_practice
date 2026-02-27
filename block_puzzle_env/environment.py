import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .logic import Board
from .pieces import PIECE_POOL
from config import REWARD, ENV


class BlockPuzzleEnv(gym.Env):
    """
    Block Puzzle окружение.

    Observation Space:
        Box(float32, shape=(4, 8, 8)) — 4 канала:
            0: игровое поле
            1-3: текущие три фигуры (в левом верхнем углу своего канала)
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
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(4, self.board_size, self.board_size),
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
        obs = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
        obs[0] = self.board.grid.astype(np.float32)
        for i, piece_idx in enumerate(self.current_pieces):
            piece = self.piece_pool[piece_idx]
            h, w = piece.shape
            obs[i + 1, :h, :w] = piece.astype(np.float32)
        return obs

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

        # --- Одновременная очистка строк и столбцов ---
        lines_cleared, is_perfect_clear = self.board.clear_lines_and_score()

        if lines_cleared > 0:
            # Combo: чем больше линий за раз, тем выгоднее
            reward += REWARD["line_cleared"] * lines_cleared * (REWARD["combo_multiplier"] ** (lines_cleared - 1))
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
