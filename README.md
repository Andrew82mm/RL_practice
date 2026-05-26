# Block Puzzle RL Agent
Разработка и исследование агента глубокого обучения с подкреплением для решения задачи дискретного размещения геометрических объектов в условиях неопределённости.

История экспериментов, архитектур и результатов: [`presentation/GENERATIONS.md`](presentation/GENERATIONS.md)

---

## Структура проекта

```
.
├── block_puzzle_env/
│   ├── __init__.py
│   ├── environment.py     # Gymnasium-среда: obs (15, 16, 16), action mask, reward
│   ├── logic.py           # Логика доски: размещение, очистка линий, BFS
│   └── pieces.py          # Пул фигур: мономино → тетрамино (все повороты)
│
├── training/
│   ├── run_training.py    # Обучение (--arch mlp/cnn/vit/spatial), resume, BC warm-start
│   ├── bc_train.py        # Behavior Cloning: сбор данных от HeuristicAgent + supervised обучение
│   ├── transfer_train.py  # Transfer learning (Gen 2 → Gen 3)
│   └── sweep.py           # Перебор гиперпараметров
│
├── evaluation/
│   ├── evaluate.py        # Оценка моделей, сравнительная таблица
│   ├── baselines.py       # RandomAgent / HeuristicAgent
│   └── statistical_tests.py  # Mann–Whitney U, Cohen's d, CLES, Bootstrap CI, Survival
│
├── utils/
│   ├── cnn_extractor.py   # HierarchicalCNN (1.29M) + SpatialCNN
│   ├── vit_extractor.py   # SmallViT (16 патчей 4×4, 2 Pre-LN слоя, 1.25M)
│   ├── logger.py          # Логирование stdout/stderr в файл
│   └── play_manual.py     # Ручная игра
│
├── visualization/
│   ├── record_game.py     # Записывает игру агента в JSON (ppo / heuristic / random)
│   └── viewer.html        # HTML-вьюер: board, pieces, heatmap логитов нейросети
│
├── notebooks/
│   ├── kaggle_train.ipynb     # BC обучение ViT на Kaggle T4 (фаза 1)
│   └── kaggle_vit_ppo.ipynb   # PPO fine-tuning ViT на Kaggle T4 (фаза 2)
│
├── models/
│   ├── gen1/              # MLP + CNN на 8×8, obs (6, 8, 8)
│   ├── gen2/              # MLP + CNN на 8×8, obs (9, 8, 8) + placement heatmap
│   ├── gen3/              # CNN на 8×8, obs (14, 8, 8) + survivability + blob + dead zones
│   └── gen4/              # CNN + ViT на 16×16, obs (15, 16, 16) + BC warm-start
│
├── results/
│   ├── charts/            # PNG-графики (кривые обучения, статистика, сравнения)
│   ├── res_eval.md        # Сводные результаты оценки
│   └── stat_test.md       # Результаты статистических тестов
│
├── report/
│   ├── report.tex                # LaTeX-отчёт по проекту
│   ├── generate_report_charts.py # Генерация графиков для отчёта (matplotlib)
│   └── visualize_diagonal.py     # Визуализация диагональной стратегии агента
│
├── presentation/
│   ├── presentation.md    # Скрипт выступления (привязан к слайдам)
│   ├── GENERATIONS.md     # История поколений агентов
│   ├── generate_charts.py # Генерация графиков (matplotlib)
│   ├── visualize_obs.py   # Визуализация 15-канального тензора наблюдений
│   ├── manim_cnn.py       # Анимационные сцены архитектуры CNN (Manim)
│   ├── tex/               # LaTeX-исходник презентации (Beamer)
│   └── media/             # Сгенерированные Manim-видео и SVG
│
├── download_checkpoints.sh  # Скачать чекпоинты Gen 4 с Kaggle (требует токен)
└── config.py              # Конфигурации: reward shaping, PPO, MLP/CNN/ViT конфиги
```

---

## Установка зависимостей

```bash
pip install stable-baselines3 sb3-contrib gymnasium torch numpy scipy tqdm
```

---

## Запуск

### Gen 1–3 (8×8, прямое PPO обучение)

Обучение:

```bash
python training/run_training.py --arch mlp --run-name mlp_v1
python training/run_training.py --arch cnn --run-name cnn_v1

python training/run_training.py --arch mlp --timesteps 2000000 --n-envs 8
python training/run_training.py --arch mlp --resume ./models/gen1/mlp_gen_1/mlp_puzzle_500000_steps.zip
```

Transfer learning (Gen 2 → Gen 3):

```bash
python training/transfer_train.py
python training/transfer_train.py --run-name cnn_gen3_v2 --timesteps 8000000 --lr 5e-5
```

Перебор гиперпараметров:

```bash
python training/sweep.py
```

### Gen 4 (16×16, двухфазный пайплайн с Behavior Cloning)

На 16×16 случайный старт PPO даёт <1 линии/эпизод — агент не получает полезного сигнала.
Решение: сначала обучить политику supervised на демонстрациях эвристики, затем дообучить PPO.

**Фаза 1 — Behavior Cloning** (локально или на Kaggle):

```bash
# Сбор 500k пар (obs, action) + supervised обучение (~7 ГБ данных, ~4 часа локально)
python training/bc_train.py --arch vit

# Только сбор данных
python training/bc_train.py --only-collect --n-samples 500000

# Только обучение (датасет уже собран)
python training/bc_train.py --arch vit --only-train \
    --dataset-path ./data/bc_dataset.npz \
    --pretrained-path ./models/vit_bc_pretrained
```

На Kaggle T4 (~2 часа): загрузи `notebooks/kaggle_train.ipynb`, добавь датасет через Add Input,
запусти через Save & Run All. Скачай `vit_bc_pretrained.zip` из вкладки Output.

**Фаза 2 — PPO fine-tuning** (локально или на Kaggle):

```bash
python training/run_training.py --arch cnn \
    --bc-pretrained ./models/bc_pretrained.zip \
    --run-name cnn_bc_v1

python training/run_training.py --arch vit \
    --bc-pretrained ./models/vit_bc_pretrained.zip \
    --run-name vit_bc_v1 --timesteps 15000000
```

На Kaggle T4 (~7 часов): загрузи `notebooks/kaggle_vit_ppo.ipynb`, добавь `vit_bc_pretrained.zip`
как Kaggle Dataset, запусти через Save & Run All.

---

## Мониторинг

```bash
tensorboard --logdir ./runs
```

---

## Оценка

```bash
# Один агент
python evaluation/evaluate.py --model ./models/gen4/cnn_bc_finetune_final.zip --n 500 --seed 42

# Сравнение нескольких + бейзлайны
python evaluation/evaluate.py \
    --models models/gen4/cnn_bc_finetune_final.zip \
             models/gen4/vit_bc_finetune_final.zip \
    --n 500 --seed 42

# Без бейзлайнов
python evaluation/evaluate.py \
    --models models/gen4/cnn_bc_finetune_final.zip \
             models/gen4/vit_bc_finetune_final.zip \
    --n 500 --seed 42 --no-baselines

# Кривая обучения по чекпоинтам
python evaluation/evaluate.py \
    --checkpoints ./models/checkpoints/ --prefix cnn_puzzle --n 100 --seed 42
```

Статистические тесты:

```bash
python evaluation/statistical_tests.py \
    --models models/gen4/cnn_bc_finetune_final.zip \
             models/gen4/vit_bc_finetune_final.zip \
    --n 500 --seed 42
```

---

## Визуализация

Запись игры в JSON:

```bash
# PPO-модель
python visualization/record_game.py \
    --agent ppo --model models/gen4/cnn_bc_finetune_final.zip \
    --n-episodes 3 --output visualization/replay_cnn.json

# Эвристика
python visualization/record_game.py \
    --agent heuristic --n-episodes 3 --output visualization/replay_heuristic.json

# Рандом
python visualization/record_game.py \
    --agent random --n-episodes 3 --output visualization/replay_random.json
```

Открой `visualization/viewer.html` в браузере, загрузи JSON через кнопку.
Горячие клавиши: `←` / `→` — шаги, `Space` — авто-воспроизведение.

---

## Ручная игра

```bash
python utils/play_manual.py
```

---

## Работа с моделями

```python
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("./models/gen4/cnn_bc_finetune_final.zip")
# norm_obs=False в run_training.py — VecNormalize не нужен при инференсе
```
