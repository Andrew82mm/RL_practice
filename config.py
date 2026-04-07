# config.py
# ----------------------------------------------------------
# Награды и штрафы (reward shaping)
# ----------------------------------------------------------
REWARD = {
    # За успешное размещение фигуры
    "place_piece": 0.05,
    # За очищенную линию (строка или столбец = 1 линия)
    "line_cleared": 3.0,
    # Множитель combo: награда умножается на кол-во линий за раз
    # итого = line_cleared * lines_count * combo_multiplier^(lines_count-1)
    "combo_multiplier": 2.0,
    # Perfect Clear — полное очищение поля
    "perfect_clear": 15.0,
    # Штраф за попытку невалидного хода (не должна срабатывать с маской)
    "invalid_move": -2.0,
    # Штраф за game over
    "game_over": -10.0,
    # Штраф за каждую занятую клетку на поле в момент game over
    "game_over_per_cell": 0.0,
}

# ----------------------------------------------------------
# Параметры окружения
# ----------------------------------------------------------
ENV = {
    "board_size": 8,
    "pieces_per_round": 3,
    # Максимальное число шагов в эпизоде
    "max_steps": 2000,
}

# ----------------------------------------------------------
# Логирование (общие пути; run_name подставляется в run_training.py)
# ----------------------------------------------------------
LOGGING = {
    "tensorboard_log": "./runs/",
    # run_name задаётся при запуске через --run-name или генерируется автоматически
    "run_name": None,
    "save_dir": "./models/",
    # Каждые N шагов сохранять чекпоинт
    "checkpoint_freq": 100_000,
    "checkpoint_dir": "./models/checkpoints/",
    # Каждые N эпизодов логировать средние метрики в TensorBoard
    "log_interval": 10,
}

# ----------------------------------------------------------
# Общие PPO-гиперпараметры (одинаковы для MLP и CNN)
# ----------------------------------------------------------
_PPO_BASE = {
    "n_envs":          8,
    "total_timesteps": 5_000_000,
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      512,
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.03,
    "vf_coef":         0.5,
    "max_grad_norm":   0.5,
}

# ----------------------------------------------------------
# MLP — политика MlpPolicy + FlattenExtractor
#
# Вход: flatten(4 x 8 x 8) = 256 значений
# Сеть: 256 -> 256 -> actor/critic головы
# ----------------------------------------------------------
MLP_TRAIN = {
    **_PPO_BASE,
    "policy":   "MlpPolicy",
    # Два скрытых слоя для actor и critic
    "net_arch": [256, 256],
    # FlattenExtractor используется по умолчанию для MlpPolicy
    "features_extractor": "flatten",
}

# ----------------------------------------------------------
# CNN — политика MlpPolicy + кастомный SmallCNN экстрактор
#
# ВАЖНО: стандартный CnnPolicy использует NatureCNN, который
# требует вход >= 36x36 и упадёт на поле 8x8.
# Решение: оставляем MlpPolicy, но передаём кастомный
# features_extractor_class через policy_kwargs.
# SmallCNN реализован в cnn_extractor.py.
#
# Архитектура SmallCNN (вход 4 x 8 x 8):
#   Conv(4->32,  3x3, pad=1) -> BN -> ReLU -> (32, 8, 8)
#   Conv(32->64, 3x3, pad=1) -> BN -> ReLU -> (64, 8, 8)
#   Conv(64->64, 3x3, pad=1) -> BN -> ReLU -> (64, 8, 8)
#   Flatten -> Linear(64*8*8 -> features_dim) -> ReLU
#
# После экстрактора идут MLP-головы из net_arch.
# ----------------------------------------------------------
CNN_TRAIN = {
    **_PPO_BASE,
    "policy":       "MlpPolicy",
    "features_dim": 256,          # выходная размерность SmallCNN
    "net_arch":     [256, 256],   # MLP-головы поверх CNN-эмбеддинга
    "features_extractor": "cnn",  # маркер: run_training подставит SmallCNN
}
