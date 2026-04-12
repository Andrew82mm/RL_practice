# Block Puzzle RL Agent
Разработка и исследование агента глубокого обучения с подкреплением для решения задачи дискретного размещения геометрических объектов в условиях неопределённости
История экспериментов, архитектур и результатов доступна в: `GENERATIONS.md`

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
├── presentatiom/
│   ├── visualize_obs.py   # Визуализация 14-канального тензора наблюдений
│   ├── manim_cnn.py       # Анимационные сцены архитектуры CNN (Manim)
│   └── genetate_charts.py # Графики и диаграммы для презентации (matplotlib)
│
│Нейросеть:
├── config.py              # Конфигурации: reward shaping, PPO, MLP/CNN
├── cnn_extractor.py       # Кастомный SmallCNN (8×8 поле)
├── run_training.py        # Обучение (MLP / CNN)
├── transfer_train.py      # Transfer learning (Gen2 → Gen3)
├── baselines.py           # RandomAgent / HeuristicAgent
│
│Оценка:
├── evaluate.py            # Оценка моделей
├── statistical_tests.py   # Статистические тесты
├── sweep.py               # Перебор гиперпараметров
├── play_manual.py         # Ручная игра
├── logger.py              # Логирование stdout/stderr
│
│Документация:
└── GENERATIONS.md         # История экспериментов
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
python run_training.py --arch mlp --run-name mlp_v1
python run_training.py --arch cnn --run-name cnn_v1

python run_training.py --arch mlp --timesteps 2000000 --n-envs 8
python run_training.py --arch mlp --resume ./models/checkpoints/mlp_puzzle_500000_steps.zip
```

Transfer learning:

```bash
python transfer_train.py
python transfer_train.py --run-name cnn_gen3_v2 --timesteps 8000000 --lr 5e-5
```

Мониторинг:

```bash
tensorboard --logdir ./runs
```

Оценка:

```bash
python evaluate.py --model ./models/mlp_v1_final.zip --n 500 --seed 42
python evaluate.py --models ./models/mlp_v1_final.zip ./models/cnn_v1_final.zip
python evaluate.py --checkpoints ./models/checkpoints/ --prefix mlp_puzzle --n 200
```

Статистика:

```bash
python statistical_tests.py --model ./models/mlp_v1_final.zip --n 500 --seed 42
```

Ручная игра:

```bash
python play_manual.py
```

---

## Работа с моделями

Загрузка:

```python
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("./models/run_name_final.zip")
```

VecNormalize:

```python
from stable_baselines3.common.vec_env import VecNormalize

vec_env = VecNormalize.load(
    "./models/run_name_vecnormalize.pkl",
    vec_env
)
```

>  `norm_obs=False` нужен только при дообучении, не при инференсе.
