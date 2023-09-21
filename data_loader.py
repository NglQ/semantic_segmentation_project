import os
import numpy as np
from PIL import Image


def load_data_from_names(root_dir: str, fnames: list, shape=(1000, 1000), mask=False, resize_image=True) -> np.array:
    # Given the root path and a list of file names as input, return the dataset
    # array.
    images = []

    for idx, img_name in enumerate(fnames):
        x = Image.open(os.path.join(root_dir, img_name))
        if resize_image:
            x = x.resize(shape)
        if not mask:
            x = np.array(x)/255.
        images.append(np.array(x))

        if idx % 100 == 99:
            print(f"Processed {idx+1} images.")

    value = np.array(images) if not mask else one_hot_encoder(np.array(images))
    return value


def one_hot_encoder(y):
    unique_values = np.unique(y)
    one_hot_encoding = np.zeros(y.shape + (len(unique_values),), dtype=np.int8)
    for i, value in enumerate(unique_values):
        one_hot_encoding[(y == value), i] = 1
    return one_hot_encoding


def get_smaller_image(image: np.array, mask: np.array):
    mask_var = np.zeros(mask.shape)
    mask.mean(axis=-1, out=mask_var)
    image_var = image[...]
    return image_var, mask_var[...]
