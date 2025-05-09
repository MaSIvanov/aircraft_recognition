from ml.config import RESIZE_VALUE, DATA_PATH, TYPES_PATH, STATIC_PATH
from ml.data_utils import (
    get_aircraft_types,
    create_training_data,
    save_training_data,
    load_training_data,
)
from ml.model import build_model, train_model, save_model
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_training_history(history, out_path):
    plt.figure(figsize=(10, 5))

    # График accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='Тренировочная точность')
    plt.plot(history.history['val_acc'], label='Проверочная точность')
    plt.title('Точность на протяжении эпох')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()

    # График loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Тренировочные потери')
    plt.plot(history.history['val_loss'], label='Проверочные потери')
    plt.title('Потери на протяжении эпох')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
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

    # Построить и сохранить график обучения
    plot_path = os.path.join(STATIC_PATH, "training_history.png")
    plot_training_history(history, plot_path)
    print(f"График обучения сохранён в {plot_path}")

    # Пример инференса (раскомментируй для теста)
    # predict_single_image(model, os.path.join(DATA_PATH, "Testing Images/Whiteman.JPG"), RESIZE_VALUE, aircraft_types)
    # test_multiple_aircraft(model, os.path.join(DATA_PATH, "Testing Images/Dyes.JPG"), RESIZE_VALUE, aircraft_types)


if __name__ == "__main__":
    main()
