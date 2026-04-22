import numpy as np
from scipy.signal import correlate2d

class Board:
    def __init__(self, size=8):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        return self.grid

    def can_place(self, piece, x, y):
        """Проверяет, можно ли разместить фигуру piece по координатам (x, y)."""
        h, w = piece.shape

        if x < 0 or y < 0 or x + w > self.size or y + h > self.size:
            return False

        target_area = self.grid[y:y + h, x:x + w]
        if np.any((target_area & piece) != 0):
            return False

        return True

    def place_piece(self, piece, x, y):
        h, w = piece.shape
        self.grid[y:y + h, x:x + w] += piece
        return self.grid

    def clear_lines_and_score(self):
        """
        Находит все заполненные строки и столбцы ОДНОВРЕМЕННО на текущем поле, затем обнуляет их все разом.
        В старой версии строки очищались первыми, из-за чего столбец мог перестать быть полностью заполненным, хотя фактически должен был очиститься вместе со строкой.
        """
        filled_rows = np.where(np.all(self.grid == 1, axis=1))[0]
        filled_cols = np.where(np.all(self.grid == 1, axis=0))[0]

        lines_cleared = len(filled_rows) + len(filled_cols)

        if len(filled_rows):
            self.grid[filled_rows, :] = 0
        if len(filled_cols):
            self.grid[:, filled_cols] = 0

        is_perfect_clear = np.sum(self.grid) == 0

        return lines_cleared, is_perfect_clear

    def has_valid_moves(self, pieces):
        """
        Проверяет, можно ли разместить хоть одну фигуру из списка матриц numpy.
        """
        for piece in pieces:
            for y in range(self.size):
                for x in range(self.size):
                    if self.can_place(piece, x, y):
                        return True
        return False

    def compute_action_mask(self, piece_pool, current_piece_indices, board_size):
        """
        Вычисляет булеву маску валидных действий для MaskablePPO.
        Размер маски: 3 * board_size * board_size.
        """
        mask = np.zeros(3 * board_size * board_size, dtype=bool)
        grid_f = self.grid.astype(np.float32)
        for slot, pool_idx in enumerate(current_piece_indices):
            piece = piece_pool[pool_idx].astype(np.float32)
            h, w = piece.shape
            # correlate2d(grid, piece) == 0 означает отсутствие перекрытия → можно поставить
            # mode='valid' даёт shape (n-h+1, n-w+1) — только позиции в пределах доски
            valid = (correlate2d(grid_f, piece, mode='valid') == 0)
            result = np.zeros((board_size, board_size), dtype=bool)
            result[:board_size - h + 1, :board_size - w + 1] = valid
            offset = slot * board_size * board_size
            mask[offset:offset + board_size * board_size] = result.ravel()
        return mask
