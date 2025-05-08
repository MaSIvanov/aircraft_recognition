from ml.config import RESIZE_VALUE, DATA_PATH, TYPES_PATH
from ml.data_utils import (
    get_aircraft_types,
    create_training_data,
    save_training_data,
    load_training_data,
)
from ml.model import build_model, train_model, save_model
import numpy as np
import os

# Получить классы
aircraft_types = get_aircraft_types()

# Создать и сохранить датасет
training_data = create_training_data(aircraft_types)
save_training_data(training_data, os.path.join(DATA_PATH, "training_data.npz"))

# Загрузить данные
training_data = load_training_data(os.path.join(DATA_PATH, "training_data.npz"))
x_train = np.array([x[0] for x in training_data]) / 255.0
y_train = np.array([x[1] for x in training_data])
input_shape = (RESIZE_VALUE, RESIZE_VALUE, 3)
num_classes = len(aircraft_types)

# Обучить модель
model = build_model(input_shape, num_classes)
model, history = train_model(model, x_train, y_train)
save_model(model, os.path.join(DATA_PATH, "aircraft_model.keras"))

# Пример инференса (раскомментируй для теста)
# predict_single_image(model, os.path.join(DATA_PATH, "Testing Images/Whiteman.JPG"), RESIZE_VALUE, aircraft_types)
# test_multiple_aircraft(model, os.path.join(DATA_PATH, "Testing Images/Dyes.JPG"), RESIZE_VALUE, aircraft_types)
