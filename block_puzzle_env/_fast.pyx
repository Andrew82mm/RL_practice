# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython-ускорение BFS для _compute_blob_and_dead_zone.

Заменяет два отдельных Python BFS прохода одним проходом на C-скорости:
  - BFS без Python-объектов (deque/list) — C-массивы + integer head/tail
  - Dead-zone check без correlate2d на компоненту — прямой перебор
    (cy, cx) × piece_offsets вместо scipy-вызова на каждую компоненту
"""
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


def compute_blob_and_dead_zone(
    np.ndarray[np.int8_t, ndim=2] grid not None,
    list valid_pos_list,
    list piece_offset_arrays,
):
    """
    Parameters
    ----------
    grid : (n, n) int8
    valid_pos_list : list of ndarray[uint8, ndim=2]
        valid_pos_list[k][py, px] = 1 если фигура k помещается в (py, px).
        Shape: (n - ph_k + 1, n - pw_k + 1)  (из correlate2d mode='valid')
    piece_offset_arrays : list of ndarray[int32, ndim=2], shape (K_k, 2)
        Клетки фигуры k: [[pr0, pc0], [pr1, pc1], ...]

    Returns
    -------
    blob_ch, dead_ch : ndarray[float32, ndim=2]
    """
    cdef int n = grid.shape[0]
    cdef int sy, sx, cy, cx
    cdef int k, idx, j
    cdef int py, px
    cdef int best_size = 0, comp_size, qhead, qtail
    cdef int is_dead, num_pieces = len(valid_pos_list)
    cdef int dead_count = 0
    cdef int vp_h, vp_w, num_offsets

    # C-массивы для BFS без Python-объектов
    cdef int* q_y = <int*>malloc(n * n * sizeof(int))
    cdef int* q_x = <int*>malloc(n * n * sizeof(int))
    cdef int* c_y = <int*>malloc(n * n * sizeof(int))
    cdef int* c_x = <int*>malloc(n * n * sizeof(int))
    cdef int* b_y = <int*>malloc(n * n * sizeof(int))
    cdef int* b_x = <int*>malloc(n * n * sizeof(int))
    cdef int* d_y = <int*>malloc(n * n * sizeof(int))
    cdef int* d_x = <int*>malloc(n * n * sizeof(int))

    cdef np.ndarray[np.uint8_t, ndim=2] empty_arr = (grid == 0).astype(np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=2] visited_arr = np.zeros((n, n), dtype=np.uint8)
    cdef np.uint8_t[:, :] empty_v = empty_arr
    cdef np.uint8_t[:, :] visited_v = visited_arr

    # Приводим к нужным типам заранее (один раз, не в цикле)
    vp_arrays  = [np.asarray(v, dtype=np.uint8)  for v in valid_pos_list]
    off_arrays = [np.asarray(o, dtype=np.int32)  for o in piece_offset_arrays]

    # Typed memoryviews — объявляем на уровне функции, переназначаем в цикле
    cdef np.uint8_t[:, :]  vp_mv
    cdef np.int32_t[:, :]  off_mv

    cdef np.ndarray[np.float32_t, ndim=2] blob_ch = np.zeros((n, n), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] dead_ch  = np.zeros((n, n), dtype=np.float32)

    try:
        for sy in range(n):
            for sx in range(n):
                if not empty_v[sy, sx] or visited_v[sy, sx]:
                    continue

                # --- BFS ---
                qhead = 0; qtail = 1; comp_size = 0
                q_y[0] = sy; q_x[0] = sx
                visited_v[sy, sx] = 1

                while qhead < qtail:
                    cy = q_y[qhead]; cx = q_x[qhead]; qhead += 1
                    c_y[comp_size] = cy; c_x[comp_size] = cx; comp_size += 1

                    if cy > 0 and empty_v[cy-1, cx] and not visited_v[cy-1, cx]:
                        visited_v[cy-1, cx] = 1
                        q_y[qtail] = cy-1; q_x[qtail] = cx; qtail += 1
                    if cy < n-1 and empty_v[cy+1, cx] and not visited_v[cy+1, cx]:
                        visited_v[cy+1, cx] = 1
                        q_y[qtail] = cy+1; q_x[qtail] = cx; qtail += 1
                    if cx > 0 and empty_v[cy, cx-1] and not visited_v[cy, cx-1]:
                        visited_v[cy, cx-1] = 1
                        q_y[qtail] = cy; q_x[qtail] = cx-1; qtail += 1
                    if cx < n-1 and empty_v[cy, cx+1] and not visited_v[cy, cx+1]:
                        visited_v[cy, cx+1] = 1
                        q_y[qtail] = cy; q_x[qtail] = cx+1; qtail += 1

                # Обновляем лучшую компоненту
                if comp_size > best_size:
                    best_size = comp_size
                    for idx in range(comp_size):
                        b_y[idx] = c_y[idx]
                        b_x[idx] = c_x[idx]

                # --- Dead-zone check ---
                # Для каждой клетки компоненты (cy, cx) и каждого смещения
                # фигуры (pr, pc): позиция размещения (py, px) = (cy-pr, cx-pc).
                # Если valid_pos[py, px]=1 → фигура влезает и задевает компоненту
                # → компонента живая.
                if num_pieces > 0:
                    is_dead = 1
                    for k in range(num_pieces):
                        if not is_dead:
                            break
                        vp_mv  = vp_arrays[k]
                        off_mv = off_arrays[k]
                        vp_h = vp_mv.shape[0]
                        vp_w = vp_mv.shape[1]
                        num_offsets = off_mv.shape[0]

                        for idx in range(comp_size):
                            if not is_dead:
                                break
                            cy = c_y[idx]; cx = c_x[idx]
                            for j in range(num_offsets):
                                py = cy - off_mv[j, 0]
                                px = cx - off_mv[j, 1]
                                if 0 <= py < vp_h and 0 <= px < vp_w and vp_mv[py, px]:
                                    is_dead = 0
                                    break

                    if is_dead:
                        for idx in range(comp_size):
                            d_y[dead_count] = c_y[idx]
                            d_x[dead_count] = c_x[idx]
                            dead_count += 1

        # Заполняем выходные массивы
        for idx in range(best_size):
            blob_ch[b_y[idx], b_x[idx]] = 1.0
        for idx in range(dead_count):
            dead_ch[d_y[idx], d_x[idx]] = 1.0

    finally:
        free(q_y); free(q_x)
        free(c_y); free(c_x)
        free(b_y); free(b_x)
        free(d_y); free(d_x)

    return blob_ch, dead_ch
