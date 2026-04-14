# Block Puzzle RL Agent
Разработка и исследование агента глубокого обучения с подкреплением для решения задачи дискретного размещения геометрических объектов в условиях неопределённости.

История экспериментов, архитектур и результатов: [`presentation/GENERATIONS.md`](presentation/GENERATIONS.md)

---

## Структура проекта

```
.
├── block_puzzle_env/
│   ├── __init__.py
│   ├── environment.py     # Gymnasium-среда: obs (14, 8, 8), action mask, reward
│   ├── logic.py           # Логика доски: размещение, очистка линий, BFS
│   └── pieces.py          # Пул фигур: мономино → тетрамино (все повороты)
│
├── training/
│   ├── run_training.py    # Обучение (MLP / CNN)
│   ├── transfer_train.py  # Transfer learning (Gen 2 → Gen 3)
│   └── sweep.py           # Перебор гиперпараметров
│
├── evaluation/
│   ├── evaluate.py        # Оценка моделей
│   ├── baselines.py       # RandomAgent / HeuristicAgent
│   └── statistical_tests.py  # Статистические тесты (Mann-Whitney, Cohen's d, CLES)
│
├── utils/
│   ├── cnn_extractor.py   # Кастомный SmallCNN (8×8 поле, stride=1)
│   ├── logger.py          # Логирование stdout/stderr
│   └── play_manual.py     # Ручная игра
│
├── models/
│   ├── gen1/
│   │   ├── mlp_gen_1/
│   │   └── cnn_gen_1/
│   ├── gen2/
│   │   ├── mlp_gen_2/
│   │   └── cnn_gen_2/
│   └── gen3/
│       ├── mlp_gen_3/
│       ├── cnn_gen_3/        # Обучение с нуля (регрессия)
│       └── cnn_gen_3_tr/     # Transfer learning из Gen 2
│
├── results/
│   ├── charts/            # PNG-графики (кривые обучения, статистика, сравнения)
│   ├── res_eval.md        # Сводные результаты оценки
│   ├── stat_test.md       # Результаты статистических тестов
│   └── sweep_results.txt  # Логи перебора гиперпараметров
│
├── presentation/
│   ├── presentation.md    # Скрипт выступления (привязан к слайдам)
│   ├── GENERATIONS.md     # История поколений агентов
│   ├── generate_charts.py # Генерация графиков (matplotlib)
│   ├── visualize_obs.py   # Визуализация 14-канального тензора наблюдений
│   ├── manim_cnn.py       # Анимационные сцены архитектуры CNN (Manim)
│   ├── tex/               # LaTeX-исходник презентации (Beamer)
│   └── media/             # Сгенерированные Manim-видео и SVG
│
├── config.py              # Конфигурации: reward shaping, PPO, MLP/CNN
└── manim.cfg              # Настройки рендера Manim
```

---

## Установка зависимостей

```bash
pip install stable-baselines3 sb3-contrib gymnasium torch numpy scipy tqdm
```

---

## Запуск

Обучение:

```bash
python training/run_training.py --arch mlp --run-name mlp_v1
python training/run_training.py --arch cnn --run-name cnn_v1

python training/run_training.py --arch mlp --timesteps 2000000 --n-envs 8
python training/run_training.py --arch mlp --resume ./models/gen1/mlp_gen_1/mlp_puzzle_500000_steps.zip
```

Transfer learning:

```bash
python training/transfer_train.py
python training/transfer_train.py --run-name cnn_gen3_v2 --timesteps 8000000 --lr 5e-5
```

Перебор гиперпараметров:

```bash
python training/sweep.py
```

Мониторинг:

```bash
tensorboard --logdir ./runs
```

Оценка:

```bash
python evaluation/evaluate.py --model ./models/gen3/mlp_gen_3/mlp_final.zip --n 500 --seed 42
python evaluation/evaluate.py --models ./models/gen3/mlp_gen_3/mlp_final.zip ./models/gen3/cnn_gen_3_tr/cnn_final.zip
python evaluation/evaluate.py --checkpoints ./models/gen1/mlp_gen_1/ --prefix mlp_puzzle --n 200
```

Статистика:

```bash
python evaluation/statistical_tests.py --model ./models/gen3/mlp_gen_3/mlp_final.zip --n 1000 --seed 42
```

Ручная игра:

```bash
python utils/play_manual.py
```

---

## Работа с моделями

Загрузка:

```python
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("./models/gen3/mlp_gen_3/mlp_final.zip")
```

VecNormalize:

```python
from stable_baselines3.common.vec_env import VecNormalize

vec_env = VecNormalize.load(
    "./models/gen3/mlp_gen_3/mlp_vecnormalize.pkl",
    vec_env
)
```

> `norm_obs=False` нужен только при дообучении, не при инференсе.
