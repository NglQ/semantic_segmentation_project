import os
from data_loader import load_data_from_names
import numpy as np
import matplotlib.pyplot as plt
import cv2


def analyze_data():
    image_names = os.listdir('./ign/images/training')
    mask_names = os.listdir('./ign/annotations/training')

    image_names.sort()
    mask_names.sort()

    y = load_data_from_names('./ign/annotations/training', mask_names, shape=(256, 256),  mask=True)

    plt.hist(np.argmax(y, axis=-1).flatten())
    plt.show()

    # for j, i in enumerate(y):
    #     if 7 in np.unique(np.argmax(i, axis=-1)):
    #         a, b = np.where(np.argmax(i, axis=-1) == 7)
    #         img = j
    #         for k, kk in zip(a,b):
    #             img[k,kk] =
    #         cv2.imshow(img)


    #for mask in y:
        # plt.hist(np.argmax(mask, axis=-1).flatten())
        # plt.show()


