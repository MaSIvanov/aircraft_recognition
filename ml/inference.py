import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def predict_single_image(model, img_path, resize_value, aircraft_types):
    image = Image.open(img_path)
    img_array = np.array(image)
    aircraft_img = cv2.resize(img_array, (resize_value, resize_value))
    patch = np.expand_dims(aircraft_img, axis=0)
    pred = model.predict(patch)
    class_idx = np.argmax(pred, axis=1)
    return aircraft_types[class_idx[0]]


def test_multiple_aircraft(model, img_path, resize_value, aircraft_types, dshow=["BareLand"], length=80, width=80,
                           step=30):
    image = Image.open(img_path)
    img_array = np.array(image)
    test_img = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR) if img_array.shape[2] == 4 else img_array
    plt.imshow(test_img)
    ax = plt.gca()
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
                        patches.Rectangle((j, i), length, width, facecolor='none', edgecolor='red', linewidth=.5))
    plt.show()
