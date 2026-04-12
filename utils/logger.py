# logger.py
import sys
import os
from datetime import datetime

class TrainingLogger:
    """
    Класс для логирования процесса обучения:
    1. Определяет архитектуру (MLP/CNN).
    2. Сохраняет гиперпараметры.
    3. Перехватывает вывод из консоли (stdout и stderr).
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.terminal = sys.stdout
        self.error_terminal = sys.stderr
        self.log_file = None
        
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def __enter__(self):
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_file:
            self.log_file.close()
        sys.stdout = self.terminal
        sys.stderr = self.error_terminal

    def write(self, message):
        """Перехват записи: пишем и в файл, и в консоль."""
        # Вывод в настоящую консоль
        self.terminal.write(message)
        
        if self.log_file:
            if '\r' in message:
                message = message.replace('\r', '').strip()
                if message: # Не пишем пустые строки
                    self.log_file.write(message + '\n')
            else:
                self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.error_terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def log_header(self):
        header = f"=== TRAINING LOG: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n"
        self.write(header)

    def log_model_info(self, model):
        """
        1. Определение и запись архитектуры.
        Анализирует класс модели и экстрактора признаков.
        """
        self.write("=== 1. ARCHITECTURE ===\n")
        
        algo_name = model.__class__.__name__
        
        # Анализируем Policy и Feature Extractor
        policy = model.policy
        extractor_class = policy.features_extractor_class
        
        # Определяем тип: MLP или CNN
        arch_type = "Unknown"
        if extractor_class.__name__ == "FlattenExtractor":
            arch_type = "MLP (Multi-Layer Perceptron)"
        elif "CNN" in extractor_class.__name__:
             arch_type = "CNN (Convolutional Neural Network)"
        
        # Считываем структуру сетей
        net_arch = model.policy_kwargs.get('net_arch', 'Default')
        
        info = (
            f"Algorithm: {algo_name}\n"
            f"Architecture Type: {arch_type}\n"
            f"Policy Class: {policy.__class__.__name__}\n"
            f"Feature Extractor: {extractor_class.__name__}\n"
            f"Network Architecture (net_arch): {net_arch}\n"
        )
        self.write(info)
        self.write("\n")

    def log_params(self, params_dict: dict):
        """
        2. Запись параметров обучения.
        """
        self.write("=== 2. TRAINING PARAMETERS ===\n")
        
        for section, params in params_dict.items():
            self.write(f"--- {section} ---\n")
            if isinstance(params, dict):
                for key, value in params.items():
                    self.write(f"{key}: {value}\n")
            else:
                self.write(str(params) + "\n")
            self.write("\n")
        
    def start_capture(self):
        """
        3. Начало перехвата консольного вывода.
        Перехватываем и stdout, и stderr.
        """
        self.write("=== 3. CONSOLE OUTPUT ===\n")
        sys.stdout = self
        sys.stderr = self

    def stop_capture(self):
        sys.stdout = self.terminal
        sys.stderr = self.error_terminal