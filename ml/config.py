import os

# Размер, до которого приводятся изображения
RESIZE_VALUE = 130

# Абсолютный путь к корню проекта (на уровень выше текущего файла)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Путь к папке data
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# Путь к папке static
STATIC_PATH = os.path.join(PROJECT_ROOT, "static")

# Путь к файлу с названиями классов
TYPES_PATH = os.path.join(DATA_PATH, "TYPE-NAMES.txt")

