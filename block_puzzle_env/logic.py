import numpy as np

class Board:
    def __init__(self, size=8):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        return self.grid

    def can_place(self, piece, x, y):
        """Проверяет, можно ли разместить фигуру piece по координатам (x, y)"""
        h, w = piece.shape
        
        # Границы поля
        if x < 0 or y < 0 or x + w > self.size or y + h > self.size:
            return False
        
        # Пересечение с существующими блоками
        # Вырезаем кусок поля, куда хотим поставить фигуру
        target_area = self.grid[y:y+h, x:x+w]
        
        # Если хоть в одной клетке пересечение (1 и 1), то запрещаем
        if np.any((target_area & piece) != 0):
            return False
            
        return True

    def place_piece(self, piece, x, y):
        h, w = piece.shape
        self.grid[y:y+h, x:x+w] += piece
        return self.grid

    def clear_lines_and_score(self):
        """
        Проверяет и очищает полные строки и столбцы
        Возвращает количество очищенных линий и флаг, стало ли поле пустым
        """
        lines_cleared = 0
        
        # Находим заполненные строки
        filled_rows = np.all(self.grid == 1, axis=1)
        if np.any(filled_rows):
            lines_cleared += np.sum(filled_rows)
            self.grid[filled_rows, :] = 0

        # Находим заполненные столбцы
        filled_cols = np.all(self.grid == 1, axis=0)
        if np.any(filled_cols):
            lines_cleared += np.sum(filled_cols)
            self.grid[:, filled_cols] = 0
            
        # Проверка на полное очищение (Perfect Clear)
        is_clear = np.sum(self.grid) == 0
        
        return lines_cleared, is_clear

    def has_valid_moves(self, pieces):
        """Проверяет, можно ли разместить хоть одну фигуру из списка"""
        for piece in pieces:
            for y in range(self.size):
                for x in range(self.size):
                    if self.can_place(piece, x, y):
                        return True
        return False