# Block Puzzle RL Agent

Агент глубокого обучения с подкреплением для решения задачи дискретного размещения геометрических объектов. Среда вдохновлена механикой Block Puzzle / Tetris: на поле 8×8 последовательно подаются наборы из 3 фигур, агент размещает их оптимально, не зная будущих фигур. При заполнении строки или столбца они исчезают. Цель — максимизировать время игры и количество очищенных линий.

---

## Структура проекта

```
.
├── block_puzzle_env/
│   ├── __init__.py
│   ├── environment.py       # Gymnasium-среда (obs, action, reward, masks)
│   ├── logic.py             # Логика доски: размещение, очистка линий, маска действий
│   └── pieces.py            # Пул фигур: мономино → тетрамино, все повороты
│
├── config.py                # Все параметры: reward shaping, PPO, MLP/CNN конфиги
├── cnn_extractor.py         # Кастомный SmallCNN для поля 8×8 (заменяет NatureCNN)
├── run_training.py          # Единая точка запуска обучения (MLP / CNN)
├── train.py                 # Устаревший запускатель (оставлен для совместимости)
├── baselines.py             # RandomAgent и HeuristicAgent (математический baseline)
├── evaluate.py              # Сравнение агентов по метрикам
├── play_manual.py           # Ручная игра через консоль
├── logger.py                # Логгер: перехват stdout/stderr + запись в файл
└── README.md
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install stable-baselines3 sb3-contrib gymnasium torch tqdm
```

### 2. Обучение

```bash
# MLP-агент (рекомендуется для начала)
python run_training.py --arch mlp --run-name mlp_v1

# CNN-агент
python run_training.py --arch cnn --run-name cnn_v1

# С переопределением параметров
python run_training.py --arch mlp --timesteps 2000000 --n-envs 4

# Продолжение обучения с чекпоинта
python run_training.py --arch mlp --resume ./models/checkpoints/mlp_puzzle_500000_steps.zip
```

Модели сохраняются в `./models/<run_name>_final.zip`.  
TensorBoard логи — в `./runs/<run_name>/`.

### 3. Мониторинг обучения

```bash
tensorboard --logdir ./runs
```

### 4. Оценка агентов

```bash
# Сравнить Random vs Heuristic (200 эпизодов)
python evaluate.py

# Добавить обученную PPO-модель
python evaluate.py --model ./models/mlp_v1_final.zip --n 500

# С фиксированным seed для воспроизводимости
python evaluate.py --model ./models/cnn_v1_final.zip --n 500 --seed 42
```

### 5. Ручная игра

```bash
python play_manual.py
# Ввод: <номер_фигуры (0-2)> <x> <y>
# Пример: 0 3 3
```

---

## Пространство наблюдений и действий

| Компонент | Описание |
|---|---|
| **Observation** | `Box(float32, 4×8×8)` — 4 канала: поле + 3 текущие фигуры |
| **Action** | `Discrete(192)` = 3 слота × 8 × 8 позиций |
| **Action Mask** | `bool[192]` — MaskablePPO никогда не выбирает невалидный ход |

Кодирование действия: `action = slot * 64 + y * 8 + x`

---

## Архитектуры агентов

### MLP

```
Observation (4×8×8)
    → Flatten → [256]
    → Linear(256→256) → ReLU
    → Linear(256→256) → ReLU
    → Actor head (softmax) / Critic head (value)
```

Вход: `flatten(4×8×8) = 256` значений. Быстро обучается, не использует spatial структуру.

### CNN

```
Observation (4×8×8)
    → Conv(4→32, 3×3, pad=1) → BN → ReLU  → (32×8×8)
    → Conv(32→64, 3×3, pad=1) → BN → ReLU → (64×8×8)
    → Conv(64→64, 3×3, pad=1) → BN → ReLU → (64×8×8)
    → Flatten → Linear(4096→256) → ReLU    → [256]
    → Linear(256→256) → ReLU
    → Actor head / Critic head
```

Обрабатывает spatial паттерны поля. Обучается медленнее, потенциально лучше планирует.

> **Почему не CnnPolicy?** Стандартный `CnnPolicy` SB3 использует `NatureCNN` (страйды 4 и 2, рассчитан на Atari 84×84), который коллапсирует 8×8 в отрицательные размерности и падает с `RuntimeError`. Решение: `SmallCNN` в `cnn_extractor.py` передаётся через `policy_kwargs`.

---

## Бейзлайны

| Агент | Описание |
|---|---|
| **RandomAgent** | Равномерно случайный выбор из валидных ходов |
| **HeuristicAgent** | Жадный алгоритм на основе 7 признаков поля (Dellacherie-style) |

Признаки `HeuristicAgent`: очищенные линии, combo-бонус, perfect clear, дыры (BFS), почти заполненные линии, компактность (adjacency), дисперсия заполнения.

---

## Reward Shaping

| Событие | Награда |
|---|---|
| Размещение фигуры | +0.05 |
| Одна очищенная линия | +3.0 |
| N линий одновременно | `3.0 × N × 2.0^(N-1)` (combo) |
| Perfect Clear | +15.0 |
| Game Over | −3.0 |
| Невалидный ход (защита) | −2.0 |

---

## Гиперпараметры PPO

| Параметр | Значение |
|---|---|
| `n_envs` | 8 (SubprocVecEnv) |
| `total_timesteps` | 5 000 000 |
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 512 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `ent_coef` | 0.03 |
| `net_arch` | [256, 256] |

---

## Зависимости

| Библиотека | Назначение |
|---|---|
| `stable-baselines3` | PPO, векторные окружения, callbacks |
| `sb3-contrib` | MaskablePPO (action masking) |
| `gymnasium` | Gym API, spaces |
| `torch` | Нейронные сети |
| `numpy` | Матричные операции |
| `tqdm` | Прогресс-бары в evaluate.py |

---

## Формат файлов моделей

Модели сохраняются в формате `.zip` (стандарт SB3):

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO.load("./models/mlp_v1_final.zip")
```

---

## TensorBoard метрики

| Метрика | Описание |
|---|---|
| `rollout/ep_rew_mean` | Средняя награда за эпизод |
| `episode/lines_cleared_mean` | Среднее очищенных линий (финал эпизода) |
| `episode/pieces_placed_mean` | Среднее размещённых фигур |
| `episode/perfect_clears_mean` | Среднее perfect clears |
| `train/entropy_loss` | Энтропия политики |
| `train/policy_gradient_loss` | Потери policy gradient |
