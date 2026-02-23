import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .logic import Board
from .pieces import PIECE_POOL

class BlockPuzzleEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.board_size = 8
        self.board = Board(self.board_size)
        
        # Пул всех возможных фигур
        self.piece_pool = PIECE_POOL
        
        # Action Space: 
        # 3 фигуры * 8 (высота) * 8 (ширина) = 192 действия
        # Действие кодируется как: action_id = piece_idx * 64 + y * 8 + x
        self.action_space = spaces.Discrete(3 * self.board_size * self.board_size)
        
        # Observation Space: 
        # 4 канала: Поле + 3 фигуры (вписанные в матрицу 8x8)
        # Значения: 0 или 1
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(4, self.board_size, self.board_size), 
            dtype=np.int8
        )
        
        self.current_pieces = [] # Индексы фигур в текущем раунде
        self.render_mode = render_mode

    def _get_obs(self):
        """
        тензор (4, 8, 8)
        Канал 0: Игровое поле
        Каналы 1-3: Текущие фигуры
        """
        obs = np.zeros((4, self.board_size, self.board_size), dtype=np.int8)
        
        # Канал 0: Поле
        obs[0] = self.board.grid
        
        # Каналы 1-3: Фигуры
        for i, piece_idx in enumerate(self.current_pieces):
            piece = self.piece_pool[piece_idx]
            h, w = piece.shape
            # Размещаем фигуру в левом верхнем углу её канала
            # Агент должен понять форму и размеры по ненулевым ячейкам
            obs[i+1, 0:h, 0:w] = piece
            
        return obs

    def _generate_valid_pieces(self):
        # Попытка генерации (с ограничением попыток, чтобы не зациклиться)
        for _ in range(100):
            indices = np.random.choice(len(self.piece_pool), size=3, replace=True)
            pieces = [self.piece_pool[i] for i in indices]
            
            # Проверяем, есть ли хоть один валидный ход
            if self.board.has_valid_moves(pieces):
                return list(indices)
        
        # Если за 100 попыток не нашли валидный набор — поле тупиковое
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.board.reset()
        self.current_pieces = self._generate_valid_pieces()
        
        if self.current_pieces is None:
            # на всякий случай
            raise RuntimeError("Не удалось сгенерировать валидные фигуры на пустом поле!")

        obs = self._get_obs()
        info = {"pieces": self.current_pieces}
        return obs, info

    def step(self, action):
        # Декодируем действие
        # action = piece_idx * 64 + y * 8 + x
        piece_idx_local = action // 64   # 0, 1 или 2
        coords = action % 64
        y = coords // 8
        x = coords % 8
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # Проверка валидности
        # Проверяем, есть ли такая фигура в наборе
        if piece_idx_local >= len(self.current_pieces):
            # Агент выбрал слот, которого нет (например, слот 3, когда есть только 2 фигуры)
            reward = -1.0 # Штраф за невалидный выбор
            return self._get_obs(), reward, terminated, truncated, {}
        
        piece_pool_idx = self.current_pieces[piece_idx_local]
        piece = self.piece_pool[piece_pool_idx]
        
        if not self.board.can_place(piece, x, y):
            # Агент выбрал позицию, куда нельзя поставить фигуру
            reward = -1.0
            return self._get_obs(), reward, terminated, truncated, {}

        # Выполняем ход
        self.board.place_piece(piece, x, y)
        
        # Очистка линий
        lines_cleared, is_clear = self.board.clear_lines_and_score()
        
        # Награда за линии (1 линия = 1 очко)
        reward += float(lines_cleared) 
        
        # Бонус за полную очистку
        if is_clear:
            reward += 10.0

        # Управление списком фигур
        # Удаляем использованную фигуру
        self.current_pieces.pop(piece_idx_local)
        
        # Если фигуры кончились — генерируем новые
        if len(self.current_pieces) == 0:
            new_pieces = self._generate_valid_pieces()
            if new_pieces is None:
                # Нельзя разместить новые фигуры то гейм овер
                terminated = True
            else:
                self.current_pieces = new_pieces

        obs = self._get_obs()
        info = {
            "lines_cleared": lines_cleared,
            "is_clear": is_clear
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print("=== FIELD ===")
            print(self.board.grid)
            print("=== CURRENT PIECES ===")
            for i, idx in enumerate(self.current_pieces):
                print(f"Piece {i} (Pool idx {idx}):")
                print(self.piece_pool[idx])
                print("---")