# Shim для обратной совместимости с сохранёнными моделями SB3.
# Модели сериализуют класс как 'cnn_extractor.SmallCNN' — этот файл
# позволяет загружать их после переноса реализации в utils/cnn_extractor.py
from utils.cnn_extractor import SmallCNN

__all__ = ["SmallCNN"]
