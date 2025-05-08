import matplotlib

matplotlib.use('Agg')  # <--- ДО импорта pyplot!

import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ml.config import RESIZE_VALUE, DATA_PATH
from ml.model import load_model
from ml.data_utils import get_aircraft_types


def detect_and_draw_aircraft(img_path, out_path, model, aircraft_types, resize_value=130,
                             length=80, width=80, step=30, dshow=["BareLand"]):
    # Открываем изображение
    image = Image.open(img_path)
    img_array = np.array(image)
    # Приводим к формату BGR если есть альфа-канал
    test_img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR) if img_array.shape[2] == 4 else img_array

    plt.figure(figsize=(12, 8))
    plt.imshow(test_img)
    ax = plt.gca()

    # Сканируем картинку патчами
    for i in range(0, test_img.shape[0], step):
        for j in range(0, test_img.shape[1], step):
            if (j + length < test_img.shape[1] and i + width < test_img.shape[0]):
                raw_patch = test_img[i:i + length, j:j + width]
                aircraft_img = cv2.resize(raw_patch, (resize_value, resize_value))
                patch = np.expand_dims(aircraft_img, axis=0)
                pred = model.predict(patch)
                ident = np.argmax(pred, axis=1)
                u = aircraft_types[ident[0]]
                if u not in dshow:
                    ax.add_patch(
                        patches.Rectangle((j, i), length, width, facecolor='none', edgecolor='red', linewidth=1.5)
                    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    img_path = r"C:\Users\Максим\PycharmProjects\aircraft_recognition\data\Dyes.jpg"
    out_path = r"C:\Users\Максим\PycharmProjects\aircraft_recognition\data\Dyes_result.jpg"

    if not os.path.exists(img_path):
        print(f"Файл {img_path} не найден.")
        return

    aircraft_types = get_aircraft_types()
    model_path = os.path.join(DATA_PATH, "aircraft_model.keras")
    if not os.path.exists(model_path):
        print(f"Модель {model_path} не найдена. Сначала обучите модель.")
        return

    model = load_model(model_path)
    detect_and_draw_aircraft(img_path, out_path, model, aircraft_types, resize_value=RESIZE_VALUE)


if __name__ == "__main__":
    main()
