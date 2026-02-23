import numpy as np

def get_all_pieces():
    """
    Возвращает список всех фигур в виде матриц numpy
    """
    pieces = []
    
    # 1. Одноклеточная (Мономино)
    pieces.append(np.array([[1]], dtype=np.int8))
    
    # 2. Двуклеточная (Домино)
    pieces.append(np.array([[1, 1]], dtype=np.int8))
    pieces.append(np.array([[1], [1]], dtype=np.int8))

    # 3. Трехклеточные (Тримино)
    # I-тримино горизонтальное
    pieces.append(np.array([[1, 1, 1]], dtype=np.int8))
    # I-тримино вертикальное
    pieces.append(np.array([[1], [1], [1]], dtype=np.int8))
    # L-тримино и его повороты
    l_tri = np.array([[1, 0], [1, 1]], dtype=np.int8)
    pieces.extend(_generate_rotations(l_tri))

    # 4. Четырехклеточные (Тетрамино) - классика
    # Квадрат (O)
    pieces.append(np.array([[1, 1], [1, 1]], dtype=np.int8))
    # T-блок
    t_block = np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int8)
    pieces.extend(_generate_rotations(t_block))
    # L-блок
    l_block = np.array([[1, 0], [1, 0], [1, 1]], dtype=np.int8)
    pieces.extend(_generate_rotations(l_block))
    # S-блок
    s_block = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8)
    pieces.extend(_generate_rotations(s_block))
    # I-блок (линия 4)
    i4_h = np.array([[1, 1, 1, 1]], dtype=np.int8)
    pieces.append(i4_h)
    i4_v = np.array([[1], [1], [1], [1]], dtype=np.int8)
    pieces.append(i4_v)

    return pieces

def _generate_rotations(matrix):
    """Генерирует уникальные повороты матрицы (0, 90, 180, 270)"""
    rots = []
    current = matrix
    for _ in range(4):
        # Проверяем, есть ли уже такая (используем tobytes для сравнения массивов)
        is_new = True
        for r in rots:
            if np.array_equal(r, current):
                is_new = False
                break
        if is_new:
            rots.append(current)
        
        # Поворот на 90 градусов против часовой стрелки
        current = np.rot90(current)
    return rots

# Создаем глобальный пул фигур при импорте
PIECE_POOL = get_all_pieces()