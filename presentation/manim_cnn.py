"""
manim_cnn.py — две презентационные сцены (v2: улучшенная наглядность)

Изменения v2:
  ObsEvolutionScene:
    + Кадр 0: каталог всех 14 каналов Gen3 (2 строки по 7)
    + Сплит-макет кадров Gen1/Gen2/Gen3: стек слева + зум-панель справа
    + Полоса эволюции каналов внизу (4 → 9 → 14)
    + Улучшена анимация heatmap: фигура скользит по доске
  CNNAnatomyScene:
    + Кадр 0: архитектурная схема SmallCNN с формами тензоров
    + Кадр 1: рост рецептивного поля 3×3 → 5×5 → 7×7
    + Показ 4 фильтров одновременно (2×2 мини-сетка)

Запуск:
    manim -pql manim_cnn.py ObsEvolutionScene
    manim -pql manim_cnn.py CNNAnatomyScene
    manim -pqh manim_cnn.py ObsEvolutionScene   # 1080p
"""

from __future__ import annotations
import numpy as np
from manim import *


# ══════════════════════════════════════════════════════════════════════
#  Общие константы
# ══════════════════════════════════════════════════════════════════════

BG          = "#0d0d1a"
CELL_FILLED = "#4a90d9"
CELL_EMPTY  = "#1a1a30"
GRID_STROKE = "#2a2a50"
GREEN_HEAT  = "#27ae60"
BLUE_BLOB   = "#2980b9"
RED_DEAD    = "#c0392b"
YELLOW_HIGH = "#f39c12"
PURPLE      = "#9b59b6"
ORANGE      = "#e67e22"

L_PIECE = np.array([[1, 0],
                    [1, 0],
                    [1, 1]], dtype=np.int8)

DEMO_BOARD = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],   # ← почти полная строка (7/8)
    [1, 1, 1, 1, 1, 1, 1, 1],   # ← полная строка
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int8)

ANATOMY_BOARD = np.array([
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1],   # ← изолированная дыра (строки 2-3, столбцы 3-4)
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0],   # ← почти полная строка
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int8)

ISOLATED_MASK = np.zeros((8, 8), dtype=np.float32)
ISOLATED_MASK[2:4, 3:5] = 1.0


# ══════════════════════════════════════════════════════════════════════
#  Общие утилиты
# ══════════════════════════════════════════════════════════════════════

def val_to_color(val: float, low: str, high: str) -> ManimColor:
    return interpolate_color(ManimColor(low), ManimColor(high), float(np.clip(val, 0, 1)))


def make_grid(
    data: np.ndarray,
    cell: float = 0.45,
    color_fn=None,
    default_empty: str = CELL_EMPTY,
    default_filled: str = CELL_FILLED,
    opacity_fn=None,
) -> VGroup:
    """VGroup из n×n квадратов. color_fn(val)→цвет или бинарный filled/empty."""
    n = data.shape[0]
    cells = VGroup()
    for r in range(n):
        for c in range(n):
            val = float(data[r, c])
            color = color_fn(val) if color_fn else (default_filled if val > 0.5 else default_empty)
            opacity = opacity_fn(val) if opacity_fn else (0.92 if val > 0.02 else 0.25)
            sq = Square(
                side_length=cell,
                fill_color=color, fill_opacity=opacity,
                stroke_color=GRID_STROKE, stroke_width=0.8,
            )
            sq.move_to(np.array([c * cell, -r * cell, 0]))
            cells.add(sq)
    cells.move_to(ORIGIN)
    return cells


def channel_thumb(data: np.ndarray, label: str, color: str,
                  cell: float = 0.21) -> VGroup:
    """Миниатюра канала для каталога (cell=0.21 → сетка ~1.7×1.7)."""
    cfn = lambda v, c=color: val_to_color(v, CELL_EMPTY, c)
    grid = make_grid(data, cell=cell, color_fn=cfn)
    lbl = Text(label, font_size=9, color=color, weight=BOLD)
    lbl.next_to(grid, DOWN, buff=0.07)
    return VGroup(grid, lbl)


def extract_activations(model_path: str, board: np.ndarray):
    """Загружает CNN-модель, прогоняет board и возвращает активации."""
    import torch
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path, device="cpu")
    cnn = model.policy.features_extractor
    cnn.eval()

    n_channels = cnn.cnn[0].weight.shape[1]
    obs = np.zeros((n_channels, 8, 8), dtype=np.float32)
    obs[0] = board.astype(np.float32)
    if n_channels > 4:
        obs[4] = board.sum(axis=1)[:, np.newaxis] / 8
    if n_channels > 5:
        obs[5] = board.sum(axis=0)[np.newaxis, :] / 8

    acts: dict[str, np.ndarray] = {}

    def hook(name):
        def _h(_, __, out):
            acts[name] = out.detach().cpu().numpy()[0]
        return _h

    cnn.cnn[1].register_forward_hook(hook("conv1"))
    cnn.cnn[3].register_forward_hook(hook("conv2"))
    cnn.cnn[5].register_forward_hook(hook("conv3"))

    x = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        cnn(x)

    return acts["conv1"], acts["conv3"]


def find_best_filter(acts: np.ndarray, score_mask: np.ndarray | None = None) -> int:
    if score_mask is not None:
        scores = (acts * score_mask[np.newaxis]).mean(axis=(1, 2))
    else:
        scores = acts.var(axis=(1, 2))
    return int(np.argmax(scores))


# ══════════════════════════════════════════════════════════════════════
#  Сцена А: Эволюция наблюдения  Gen1 → Gen2 → Gen3  (v3)
#
#  Дизайн:
#    • Один крупный grid по центру (CELL=0.65, 8×8 ≈ 5.2×5.2 ед.)
#    • Два слоя: base (доска, z=1) + overlay (данные канала, z=2)
#    • Переходы = анимация цветов ячеек, не FadeOut/FadeIn объектов
#    • Нарратив: вопрос над сеткой → матрица → ответ в подписи
#    • Трекер внизу: 14 цветных точек (не мини-сетки)
#    • Пульсация для blob и dead zones
# ══════════════════════════════════════════════════════════════════════

class ObsEvolutionScene(Scene):
    """
    Дизайн: Вопрос → Матрица → Ответ.

    Один крупный grid (8×8, CELL=0.65) по центру на протяжении всей сцены.
    Два постоянных слоя:
      base_cells  (z=1) — статическая доска (синие/тёмные клетки)
      overlay_cells (z=2) — данные канала, меняем цвета, не объект
    UI (z=3): вопрос сверху, подпись канала снизу, 14 точек трекера внизу.
    Переходы: AnimationGroup по 64 ячейкам, FadeTransform для текста.
    Пульсация: blob (синий) и dead zone (красный), 3 цикла.
    """

    CELL = 0.65   # 8 × 0.65 = 5.2 ед. — читаемо, умещается в кадр

    def construct(self):
        self.camera.background_color = BG
        self._precompute()
        self._build_grid_layers()
        self._build_tracker()

        self._act_intro()
        self._act_ch0_board()
        self._act_ch1_piece()
        self._act_ch4_fill()
        self._act_gen1_gap()
        self._act_ch6_heatmap()
        self._act_ch9_survivability()
        self._act_ch12_blob()
        self._act_ch13_dead()

    # ══ Предвычисление ══════════════════════════════════════════════

    def _precompute(self):
        board = DEMO_BOARD
        piece = L_PIECE

        ch1 = np.zeros((8, 8), dtype=float)
        h, w = piece.shape
        ch1[:h, :w] = piece
        self._ch1 = ch1

        # Ch 4: row fill (broadcast по строкам)
        self._ch4 = np.outer(board.sum(axis=1) / 8, np.ones(8))
        # Ch 5: col fill (broadcast по столбцам)
        self._ch5 = np.outer(np.ones(8), board.sum(axis=0) / 8)

        ph, pw = piece.shape
        hm = np.zeros((8, 8), dtype=float)
        self._valid = []
        for r in range(8 - ph + 1):
            for c in range(8 - pw + 1):
                if np.all(board[r:r+ph, c:c+pw] + piece <= 1):
                    hm[r, c] = 1.0
                    self._valid.append((r, c))
        self._ch6 = hm

        # Ch 9: survivability — для каждой допустимой позиции piece1
        # считаем, сколько позиций остаётся для piece2 (2×2 квадрат)
        self._ch9 = self._compute_surv_demo(board, piece)

        self._ch12 = self._largest_blob(board).astype(float)
        self._ch13 = ((board == 0).astype(float)) * (1.0 - self._ch12)

    def _compute_surv_demo(self, board: np.ndarray, piece: np.ndarray) -> np.ndarray:
        """Survivability: для каждой позиции piece1 — доля позиций piece2 которые остаются."""
        piece2 = np.array([[1, 1], [1, 1]], dtype=np.int8)  # 2×2 квадрат (piece2 в демо)
        p2h, p2w = piece2.shape
        ph, pw = piece.shape

        # Сколько всего позиций piece2 на чистой доске (до хода)
        total2 = sum(
            1 for r2 in range(8 - p2h + 1) for c2 in range(8 - p2w + 1)
            if np.all(board[r2:r2+p2h, c2:c2+p2w] + piece2 <= 1)
        )

        surv = np.zeros((8, 8), dtype=float)
        for r, c in self._valid:
            temp = board.copy()
            temp[r:r+ph, c:c+pw] += piece
            # Очищаем заполненные строки и столбцы (как в реальной игре)
            filled_rows = [i for i in range(8) if temp[i, :].sum() == 8]
            filled_cols = [j for j in range(8) if temp[:, j].sum() == 8]
            for i in filled_rows:
                temp[i, :] = 0
            for j in filled_cols:
                temp[:, j] = 0
            count2 = sum(
                1 for r2 in range(8 - p2h + 1) for c2 in range(8 - p2w + 1)
                if np.all(temp[r2:r2+p2h, c2:c2+p2w] + piece2 <= 1)
            )
            surv[r, c] = float(count2)
        # Нормируем по максимуму чтобы получить градиент [0,1]
        max_val = surv.max()
        if max_val > 0:
            surv = surv / max_val
        return surv

    # ══ Построение постоянных слоёв ══════════════════════════════════

    def _build_grid_layers(self):
        """
        base_cells  (z=1): показывают доску (синие/тёмные), не меняются.
        overlay_cells (z=2): данные канала поверх, меняем fill в каждом акте.
        Оба слоя добавляются в сцену, но изначально невидимы (opacity=0).
        """
        # Центр сетки — чуть выше середины, чтоб вместился трекер внизу
        center = UP * 0.15

        self._base_cells: list = []
        self._overlay_cells: list = []
        base_grp = VGroup()
        overlay_grp = VGroup()

        for r in range(8):
            for c in range(8):
                filled = bool(DEMO_BOARD[r, c])
                # Base: статическая доска
                bsq = Square(
                    side_length=self.CELL,
                    fill_color=CELL_FILLED if filled else CELL_EMPTY,
                    fill_opacity=0.0,          # начинаем невидимыми
                    stroke_color=GRID_STROKE,
                    stroke_width=0.9,
                )
                bsq.move_to(center + RIGHT * (c - 3.5) * self.CELL
                                   + UP   * (3.5 - r) * self.CELL)
                bsq.set_z_index(1)
                self._base_cells.append(bsq)
                base_grp.add(bsq)

                # Overlay: данные канала (изначально полностью прозрачный)
                osq = Square(
                    side_length=self.CELL,
                    fill_color=WHITE,
                    fill_opacity=0.0,
                    stroke_width=0.0,
                )
                osq.move_to(bsq.get_center())
                osq.set_z_index(2)
                self._overlay_cells.append(osq)
                overlay_grp.add(osq)

        self._base_grp = base_grp
        self._overlay_grp = overlay_grp
        # Добавляем в сцену (невидимые, но готовые к анимации)
        self.add(base_grp, overlay_grp)

    def _build_tracker(self):
        """
        14 каналов внизу экрана, сгруппированные по логике.
        Каждый пункт: цветная точка + маленький ярлык снизу.
        Над каждой группой — категория.
        Прошлые: приглушённые. Текущий: яркий. Будущие: почти невидимые.
        """
        # (ярлык, цвет, категория | None=продолжение группы)
        TRACKER_META = [
            ("Д",   CELL_FILLED,  "Доска"),
            ("Ф1",  ORANGE,       "Фигуры"),
            ("Ф2",  ORANGE,       None),
            ("Ф3",  ORANGE,       None),
            ("Ст",  YELLOW_HIGH,  "Заполн."),
            ("Кл",  YELLOW_HIGH,  None),
            ("Х1",  GREEN_HEAT,   "Хитмапы"),
            ("Х2",  GREEN_HEAT,   None),
            ("Х3",  GREEN_HEAT,   None),
            ("S1",  PURPLE,       "Выжив."),
            ("S2",  PURPLE,       None),
            ("S3",  PURPLE,       None),
            ("Bl",  BLUE_BLOB,    "Топол."),
            ("Мр",  RED_DEAD,     None),
        ]

        self._tracker_dots: list = []
        items_grp = VGroup()

        for lbl, color, _group in TRACKER_META:
            dot = Dot(radius=0.075, color=color, fill_opacity=0.12)
            dot.set_z_index(3)
            ch_lbl = Text(lbl, font_size=8, color=color)
            ch_lbl.next_to(dot, DOWN, buff=0.04)
            item = VGroup(dot, ch_lbl)
            items_grp.add(item)
            self._tracker_dots.append(dot)

        # Между группами чуть больший отступ
        group_starts = [i for i, (_, _, g) in enumerate(TRACKER_META) if g is not None]
        items_grp.arrange(RIGHT, buff=0.18)

        # Увеличиваем зазор перед каждой новой группой (кроме первой)
        for gi in group_starts[1:]:
            items_grp[gi].shift(RIGHT * 0.14)
            for j in range(gi + 1, len(TRACKER_META)):
                items_grp[j].shift(RIGHT * 0.14)

        items_grp.to_edge(DOWN, buff=0.10)

        # Группировые подписи над первым элементом каждой группы
        prev_start = None
        group_labels = VGroup()
        group_colors = {}
        for i, (_, color, grp) in enumerate(TRACKER_META):
            if grp is not None:
                group_colors[i] = (grp, color)
        sorted_starts = sorted(group_colors.keys())

        for idx, gs in enumerate(sorted_starts):
            grp_name, grp_color = group_colors[gs]
            # Найдём последний элемент этой группы
            if idx + 1 < len(sorted_starts):
                ge = sorted_starts[idx + 1] - 1
            else:
                ge = len(TRACKER_META) - 1
            x_start = items_grp[gs].get_center()[0]
            x_end   = items_grp[ge].get_center()[0]
            mid_x   = (x_start + x_end) / 2
            top_y   = items_grp[gs][0].get_center()[1] + 0.22
            glbl = Text(grp_name, font_size=8, color=grp_color, slant=ITALIC)
            glbl.move_to([mid_x, top_y, 0])
            glbl.set_z_index(3)
            glbl.set_fill(opacity=0.55)
            group_labels.add(glbl)

        self.add(items_grp, group_labels)
        self._tracker_grp = VGroup(items_grp, group_labels)

        # Плейсхолдеры для текстовых UI-элементов
        self._question_mob = None
        self._caption_mob  = None

    # ══ Механизм переходов ═══════════════════════════════════════════

    def _set_question(self, text: str, color: str = WHITE):
        """Вопрос над сеткой. Первый вызов — FadeIn, следующие — FadeTransform."""
        new_q = Text(text, font_size=26, color=color, weight=BOLD)
        new_q.set_z_index(3)
        new_q.next_to(self._base_grp, UP, buff=0.28)

        if self._question_mob is None:
            self.play(FadeIn(new_q), run_time=0.45)
        else:
            self.play(FadeTransform(self._question_mob, new_q), run_time=0.50)
        self._question_mob = new_q

    def _set_caption(self, ch_name: str, desc: str, color: str = WHITE):
        """Подпись канала под сеткой. Плавный FadeTransform."""
        name_lbl = Text(ch_name, font_size=19, color=color, weight=BOLD)
        desc_lbl = Text(desc,    font_size=14, color=GREY)
        new_cap = VGroup(name_lbl, desc_lbl).arrange(DOWN, buff=0.07, aligned_edge=LEFT)
        new_cap.set_z_index(3)
        new_cap.next_to(self._base_grp, DOWN, buff=0.24)
        new_cap.align_to(self._base_grp, LEFT)

        if self._caption_mob is None:
            self.play(FadeIn(new_cap), run_time=0.40)
        else:
            self.play(FadeTransform(self._caption_mob, new_cap), run_time=0.45)
        self._caption_mob = new_cap

    def _show_overlay(self, data: np.ndarray, color: str, run_time: float = 0.70):
        """
        Плавно окрашивает overlay-слой в данные канала.
        Пустые ячейки (val≤0.02) становятся полностью прозрачными.
        """
        anims = []
        for r in range(8):
            for c in range(8):
                val = float(data[r, c])
                idx = r * 8 + c
                if val > 0.02:
                    tgt_color   = val_to_color(val, CELL_EMPTY, color)
                    tgt_opacity = 0.55 + val * 0.35
                else:
                    tgt_color   = WHITE
                    tgt_opacity = 0.0
                anims.append(
                    self._overlay_cells[idx].animate.set_fill(tgt_color, tgt_opacity)
                )
        self.play(AnimationGroup(*anims), run_time=run_time)

    def _clear_overlay(self, run_time: float = 0.40):
        anims = [c.animate.set_fill(WHITE, 0.0) for c in self._overlay_cells]
        self.play(AnimationGroup(*anims), run_time=run_time)

    def _light_tracker(self, active: int):
        """Активирует точку active, приглушает прошлые, гасит будущие."""
        anims = []
        for i, dot in enumerate(self._tracker_dots):
            if i < active:
                anims.append(dot.animate.set_fill(opacity=0.45))
            elif i == active:
                anims.append(dot.animate.set_fill(opacity=1.00))
            else:
                anims.append(dot.animate.set_fill(opacity=0.10))
        self.play(AnimationGroup(*anims), run_time=0.30)

    def _pulse(self, cells: list, color: str, n: int = 3):
        """Пульсация выбранных ячеек overlay (привлекает взгляд)."""
        for _ in range(n):
            self.play(
                *[c.animate.set_fill(color, 0.85) for c in cells],
                run_time=0.28, rate_func=rush_into,
            )
            self.play(
                *[c.animate.set_fill(color, 0.32) for c in cells],
                run_time=0.28, rate_func=rush_from,
            )

    # ══ Акты ══════════════════════════════════════════════════════════

    def _act_intro(self):
        """Титул + плавное появление доски."""
        title = Text("Block Puzzle — что видит агент?",
                     font_size=30, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.20)
        title.set_z_index(3)

        # Анимируем появление base-слоя — ячейки за ячейкой (LaggedStart)
        self.play(FadeIn(title), run_time=0.5)
        self.play(
            LaggedStart(
                *[c.animate.set_fill(
                    opacity=0.88 if DEMO_BOARD[i // 8, i % 8] else 0.30
                  ) for i, c in enumerate(self._base_cells)],
                lag_ratio=0.01,
            ),
            run_time=0.9,
        )
        self.wait(0.7)
        self.play(FadeOut(title), run_time=0.35)

    def _act_ch0_board(self):
        """«Где уже стоят блоки?» — вспышка заполненных клеток."""
        # Gen 1 badge — остаётся до _act_gen1_gap
        self._gen1_badge = Text("Gen 1", font_size=18, color=CELL_FILLED, weight=BOLD)
        self._gen1_badge.set_z_index(3)
        self._gen1_badge.to_corner(UR, buff=0.30)
        self.play(FadeIn(self._gen1_badge), run_time=0.30)

        self._set_question("Где уже стоят блоки?")
        self._light_tracker(0)

        filled = [self._base_cells[r * 8 + c]
                  for r in range(8) for c in range(8) if DEMO_BOARD[r, c]]
        # Яркая вспышка → возврат
        self.play(*[c.animate.set_fill(CELL_FILLED, 1.0) for c in filled],
                  run_time=0.35)
        self.play(*[c.animate.set_fill(CELL_FILLED, 0.88) for c in filled],
                  run_time=0.30)

        self._set_caption("Ch 0: Board", "Бинарная маска занятых клеток")
        self.wait(2.0)

    def _act_ch1_piece(self):
        """«Какая у меня фигура?» — оранжевый overlay 3 ячеек L-фигуры."""
        self._set_question("Какая у меня фигура?")
        self._light_tracker(1)
        self._show_overlay(self._ch1, ORANGE, run_time=0.55)

        # Обводим ячейки фигуры
        piece_cells = [self._overlay_cells[r * 8 + c]
                       for r in range(8) for c in range(8) if self._ch1[r, c] > 0]
        box = SurroundingRectangle(
            VGroup(*piece_cells), color=ORANGE, stroke_width=2.5, buff=0.04
        )
        box.set_z_index(3)
        self.play(Create(box), run_time=0.45)
        self._set_caption("Ch 1: Piece shape",
                          "Форма фигуры закодирована в левом верхнем углу")
        self.wait(2.0)
        self.play(FadeOut(box), run_time=0.3)

    def _act_ch4_fill(self):
        """«Насколько заполнены строки и столбцы?» — горизонтальные и вертикальные полосы."""
        self._set_question("Насколько заполнены строки и столбцы?", color=YELLOW_HIGH)
        self._light_tracker(4)

        # Ch 4: строки — горизонтальные полосы
        self._show_overlay(self._ch4, YELLOW_HIGH, run_time=0.65)
        self._set_caption("Ch 4: Row Fill",
                          "Каждая строка закрашена пропорционально своей заполненности",
                          color=YELLOW_HIGH)
        self.wait(1.8)

        # Ch 5: столбцы — вертикальные полосы
        self._light_tracker(5)
        self._show_overlay(self._ch5, ORANGE, run_time=0.65)
        self._set_caption("Ch 5: Col Fill",
                          "Каждый столбец закрашен пропорционально своей заполненности",
                          color=ORANGE)
        self.wait(1.8)

        # Строка 6 почти полная → самая яркая
        hot_row = [self._overlay_cells[6 * 8 + c] for c in range(8)]
        self.play(*[c.animate.set_fill(YELLOW_HIGH, 1.0) for c in hot_row],
                  run_time=0.30, rate_func=rush_into)
        self.play(*[c.animate.set_fill(YELLOW_HIGH, 0.82) for c in hot_row],
                  run_time=0.25, rate_func=rush_from)
        self.wait(0.8)
        self._clear_overlay(run_time=0.40)

    def _act_ch9_survivability(self):
        """«Стоит ли рисковать?» — карта выживаемости после хода."""
        self._set_question("Если я поставлю сюда — выживу ли?", color=PURPLE)
        self._light_tracker(9)

        # Показываем карту survivability: зелёный = безопасно, тёмный = риск
        self._show_overlay(self._ch9, GREEN_HEAT, run_time=0.80)
        self._set_caption("Ch 9: Survivability",
                          "Доля позиций следующей фигуры, которые остаются после хода",
                          color=PURPLE)

        # Лучшая позиция — обводка
        valid_scores = [(self._ch9[r, c], r, c) for r, c in self._valid]
        if valid_scores:
            best_score, br, bc = max(valid_scores)
            worst_score = min(s for s, _, _ in valid_scores)
            # Пороги относительны: топ 20% = "безопасно", нижние 20% = "риск"
            spread = best_score - worst_score
            thresh_safe  = best_score  - spread * 0.20
            thresh_risky = worst_score + spread * 0.20

            safe  = [(r, c) for _, r, c in valid_scores if self._ch9[r, c] >= thresh_safe]
            risky = [(r, c) for _, r, c in valid_scores if self._ch9[r, c] <= thresh_risky]

            if safe:
                safe_cells = [self._overlay_cells[r * 8 + c] for r, c in safe]
                box_safe = SurroundingRectangle(
                    VGroup(*safe_cells) if len(safe_cells) > 1 else safe_cells[0],
                    color=GREEN_HEAT, stroke_width=2.2, buff=0.06,
                )
                box_safe.set_z_index(3)
                self.play(Create(box_safe), run_time=0.40)
            else:
                box_safe = None

            if risky:
                risky_cells = [self._overlay_cells[r * 8 + c] for r, c in risky]
                self.play(*[c.animate.set_fill(RED_DEAD, 0.72) for c in risky_cells],
                          run_time=0.40)
        else:
            box_safe = None

        self.wait(2.0)

        if box_safe:
            self.play(FadeOut(box_safe), run_time=0.30)
        self._clear_overlay(run_time=0.40)

    def _act_gen1_gap(self):
        """Gen 1 не мог ответить — экран пустеет, появляется вопросительный знак."""
        self._set_question("Куда я могу поставить фигуру?", color=YELLOW_HIGH)
        self._clear_overlay(run_time=0.50)
        self._light_tracker(6)   # будущий канал уже подсвечен

        qmark = Text("?", font_size=110, color=GREY, weight=BOLD)
        qmark.set_z_index(3)
        qmark.move_to(UP * 0.15)

        cant = Text("Gen 1 не мог ответить на этот вопрос.",
                    font_size=17, color="#e74c3c")
        cant.set_z_index(3)
        cant.next_to(self._base_grp, DOWN, buff=0.24)

        fade_out = [FadeIn(qmark)]
        if self._caption_mob:
            fade_out.append(FadeOut(self._caption_mob))
            self._caption_mob = None
        if hasattr(self, "_gen1_badge") and self._gen1_badge:
            fade_out.append(FadeOut(self._gen1_badge))
            self._gen1_badge = None
        self.play(*fade_out, run_time=0.45)

        self.play(FadeIn(cant), run_time=0.4)
        self.wait(1.5)
        self.play(FadeOut(qmark), FadeOut(cant), run_time=0.35)

    def _act_ch6_heatmap(self):
        """Момент «Ага!»: допустимые позиции загораются одна за другой."""
        gen2_badge = Text("Gen 2", font_size=18, color=GREEN_HEAT, weight=BOLD)
        gen2_badge.set_z_index(3)
        gen2_badge.to_corner(UR, buff=0.30)
        self.play(FadeIn(gen2_badge), run_time=0.30)

        # Зажигаем клетки допустимых позиций одну за другой
        valid_cells = [self._overlay_cells[r * 8 + c] for r, c in self._valid]
        self.play(
            LaggedStart(
                *[c.animate.set_fill(GREEN_HEAT, 0.88) for c in valid_cells],
                lag_ratio=0.07,
            ),
            run_time=1.3,
        )

        self._set_caption("Ch 6: Heatmap 1",
                          "Карта всех допустимых позиций для фигуры", color=GREEN_HEAT)
        self.wait(2.2)
        self.play(FadeOut(gen2_badge), run_time=0.3)
        self._clear_overlay(run_time=0.45)

    def _act_ch12_blob(self):
        """«Где моя территория?» — синий blob, затем пульсирует."""
        self._set_question("Где ещё есть место для манёвра?", color=BLUE_BLOB)

        gen3_badge = Text("Gen 3", font_size=18, color=PURPLE, weight=BOLD)
        gen3_badge.set_z_index(3)
        gen3_badge.to_corner(UR, buff=0.30)
        self.play(FadeIn(gen3_badge), run_time=0.30)

        self._light_tracker(12)
        self._show_overlay(self._ch12, BLUE_BLOB, run_time=0.80)
        self._set_caption("Ch 12: Blob map",
                          "Крупнейшая связная пустая область", color=BLUE_BLOB)

        blob_cells = [self._overlay_cells[r * 8 + c]
                      for r in range(8) for c in range(8) if self._ch12[r, c] > 0]
        self._pulse(blob_cells, BLUE_BLOB, n=2)
        self.wait(0.8)
        self.play(FadeOut(gen3_badge), run_time=0.3)

    def _act_ch13_dead(self):
        """«Куда точно не попасть?» — красный overlay поверх синего blob."""
        self._set_question("Куда я точно не попаду?", color=RED_DEAD)
        self._light_tracker(13)

        # Blob остаётся синим; мёртвые зоны — другие пустые клетки
        dead_cells = [self._overlay_cells[r * 8 + c]
                      for r in range(8) for c in range(8) if self._ch13[r, c] > 0]
        self.play(
            *[c.animate.set_fill(RED_DEAD, 0.82) for c in dead_cells],
            run_time=0.60,
        )
        self._set_caption("Ch 13: Dead zones",
                          "Клетки, куда ни одна из текущих фигур не попадёт",
                          color=RED_DEAD)
        # Агрессивная пульсация — привлекаем внимание к проблеме
        self._pulse(dead_cells, RED_DEAD, n=3)
        self.wait(1.5)

    # ══ Утилиты ══════════════════════════════════════════════════════

    def _largest_blob(self, board: np.ndarray) -> np.ndarray:
        from collections import deque
        empty = (board == 0)
        vis   = np.zeros((8, 8), dtype=bool)
        best: list = []
        for sy in range(8):
            for sx in range(8):
                if not empty[sy, sx] or vis[sy, sx]:
                    continue
                comp: list = []
                q: deque = deque([(sy, sx)])
                vis[sy, sx] = True
                while q:
                    cy, cx = q.popleft()
                    comp.append((cy, cx))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < 8 and 0 <= nx < 8 and empty[ny, nx] and not vis[ny, nx]:
                            vis[ny, nx] = True
                            q.append((ny, nx))
                if len(comp) > len(best):
                    best = comp
        ch = np.zeros((8, 8), dtype=np.float32)
        for y, x in best:
            ch[y, x] = 1.0
        return ch


# ══════════════════════════════════════════════════════════════════════
#  Сцена Б: Анатомия SmallCNN
# ══════════════════════════════════════════════════════════════════════

MODEL_PATH = "./models/cnn_2_final.zip"


class CNNAnatomyScene(Scene):
    """
    Показывает слои SmallCNN (Input → Conv2d × 3 → Flatten → Linear)
    с описанием каждого слоя. Без примеров свёрток и активаций.
    """

    def construct(self):
        self.camera.background_color = BG

        title = Text("SmallCNN — архитектура экстрактора признаков",
                     font_size=26, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title), run_time=0.4)

        # (label, shape, color, description)
        layers = [
            (
                "Input",
                "(C, 8, 8)",
                CELL_FILLED,
                "Входной тензор: доска 8×8, C бинарных каналов.\n"
                "Канал 0 — занятые клетки, каналы 1–N — где лежат фигуры,\n"
                "последний канал — текущий падающий блок и его позиция.\n"
                "Бинарное представление: 1.0 = занято, 0.0 = пусто.",
            ),
            (
                "Conv2d\n3×3 × 32\n+ ReLU",
                "(32, 8, 8)",
                GREEN_HEAT,
                "32 фильтра 3×3, padding=1, stride=1 → spatial размер не меняется.\n"
                "Каждый фильтр выучивает локальный паттерн: горизонталь, вертикаль,\n"
                "угол, плотность заполнения. ReLU убивает отрицательные активации.\n"
                "Рецептивное поле = 3×3 пикселя входной доски.",
            ),
            (
                "Conv2d\n3×3 × 64\n+ ReLU",
                "(64, 8, 8)",
                GREEN_HEAT,
                "64 фильтра 3×3 — комбинирует признаки первого слоя.\n"
                "Видит сочетания локальных паттернов: «два занятых ряда рядом»,\n"
                "«угол + горизонталь». Рецептивное поле расширяется до 5×5.\n"
                "Удвоение каналов позволяет кодировать более сложные структуры.",
            ),
            (
                "Conv2d\n3×3 × 64\n+ ReLU",
                "(64, 8, 8)",
                GREEN_HEAT,
                "Третий свёрточный слой, рецептивное поле = 7×7 —\n"
                "почти вся доска видна из любой точки.\n"
                "Фильтры кодируют высокоуровневые концепции: мёртвые зоны,\n"
                "изолированные полости, плотность заполнения строк и столбцов.",
            ),
            (
                "Flatten",
                "(4096)",
                YELLOW_HIGH,
                "Разворачивает тензор 64 × 8 × 8 = 4096 в одномерный вектор.\n"
                "Никаких обучаемых параметров — чисто геометрическая операция.\n"
                "После этого пространственная структура потеряна: полносвязный\n"
                "слой работает с вектором признаков, а не с картой активаций.",
            ),
            (
                "Linear\n→ 256\n+ ReLU",
                "(256)",
                ORANGE,
                "Полносвязный слой: 4096 × 256 + 256 = 1 049 856 параметров.\n"
                "Сжимает вектор признаков в компактное представление размером 256.\n"
                "Выход этого слоя — вход для Policy Head и Value Head агента.\n"
                "ReLU сохраняет нелинейность перед финальными головами сети.",
            ),
        ]

        # ── Строим блоки ──────────────────────────────────────────
        blocks = VGroup()
        shape_labels = VGroup()

        for name, shape, color, _ in layers:
            rect = RoundedRectangle(
                corner_radius=0.15,
                width=1.90, height=1.45,
                fill_color=color, fill_opacity=0.18,
                stroke_color=color, stroke_width=2.4,
            )
            lbl = Text(name, font_size=13, color=color, weight=BOLD)
            lbl.move_to(rect)
            shape_lbl = Text(shape, font_size=11, color=GREY)
            shape_lbl.next_to(rect, DOWN, buff=0.09)
            blocks.add(VGroup(rect, lbl))
            shape_labels.add(shape_lbl)

        blocks.arrange(RIGHT, buff=0.30)
        blocks.move_to(UP * 1.65)

        for i, sh in enumerate(shape_labels):
            sh.next_to(blocks[i], DOWN, buff=0.09)

        arrows = VGroup()
        for i in range(len(blocks) - 1):
            arr = Arrow(
                blocks[i].get_right(), blocks[i + 1].get_left(),
                buff=0.05, color=GREY, stroke_width=1.8,
                max_tip_length_to_length_ratio=0.25,
            )
            arrows.add(arr)

        self.play(
            LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.12),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.10),
            run_time=0.7,
        )
        self.play(
            LaggedStart(*[FadeIn(s) for s in shape_labels], lag_ratio=0.10),
            run_time=0.5,
        )

        # ── Описания появляются по очереди ────────────────────────
        desc_box = RoundedRectangle(
            corner_radius=0.15,
            width=12.5, height=2.80,
            fill_color="#12122a", fill_opacity=0.95,
            stroke_color=GREY, stroke_width=1.3,
        )
        desc_box.to_edge(DOWN, buff=0.15)

        self.play(FadeIn(desc_box), run_time=0.3)

        SCALE_UP = 1.22
        prev_desc = None
        prev_block_idx = None

        for i, (_, _, color, desc_text) in enumerate(layers):
            desc = Text(desc_text, font_size=18, color=WHITE, line_spacing=1.4)
            desc.move_to(desc_box)

            anims = [FadeIn(desc)]

            # Увеличиваем текущий блок + shape label
            anims += [
                blocks[i].animate.scale(SCALE_UP),
                shape_labels[i].animate.scale(SCALE_UP),
            ]
            # Возвращаем предыдущий
            if prev_block_idx is not None:
                anims += [
                    blocks[prev_block_idx].animate.scale(1 / SCALE_UP),
                    shape_labels[prev_block_idx].animate.scale(1 / SCALE_UP),
                ]
            if prev_desc:
                anims += [FadeOut(prev_desc)]

            self.play(*anims, run_time=0.5)
            self.wait(7.0)

            prev_desc = desc
            prev_block_idx = i

        self.wait(1.0)
