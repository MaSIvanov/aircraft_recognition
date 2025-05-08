import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from ml.config import RESIZE_VALUE, DATA_PATH, TYPES_PATH


def image_process(plane="B-1"):
    datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0.0,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    selected_path = os.path.join(DATA_PATH, plane)
    for img_ in os.listdir(selected_path):
        img = load_img(os.path.join(selected_path, img_))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=selected_path, save_prefix=img_, save_format='jpeg'):
            i += 1
            if i > 20:
                break


def get_aircraft_types():
    with open(TYPES_PATH, "r") as f:
        return [t.replace('"', '') for t in f.read().split()]


def create_training_data(aircraft_types):
    training_data = []
    for type_ in aircraft_types:
        selected_path = os.path.join(DATA_PATH, type_)
        class_num = aircraft_types.index(type_)
        for img in os.listdir(selected_path):
            try:
                image = Image.open(os.path.join(selected_path, img))
                img_array = np.array(image)
                aircraft_img = cv2.resize(img_array, (RESIZE_VALUE, RESIZE_VALUE))
                training_data.append([aircraft_img, class_num])
            except Exception as e:
                print("error:", selected_path, img)
    return training_data


def save_training_data(training_data, filename):
    np.savez_compressed(filename, a=training_data, dtype=object)


def load_training_data(filename):
    loaded = np.load(filename, allow_pickle=True)
    training_data = loaded['a']
    for x in range(len(training_data)):
        if training_data[x][0].shape[2] == 4:
            training_data[x][0] = cv2.cvtColor(training_data[x][0], cv2.COLOR_BGRA2BGR)
    return training_data
