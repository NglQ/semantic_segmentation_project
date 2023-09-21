# Utilities
import os
import datetime
from data_loader import load_data_from_names
from glob import glob

# Algebra
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Neural Networks
import tensorflow as tf
from tensorflow import keras as ks
from nn_builder import build_cnn, get_model, mobileunet
from callbacks import DisplayCallback
from keras.optimizers import Adam
import random
from custom_metrics import dice_coef_multilabel, np_dice_coef_multilabel


def test_model():
    image_names = os.listdir('./ign_new/images/validation')
    mask_names = os.listdir('./ign_new/annotations/validation')

    image_names.sort()
    mask_names.sort()

    x = load_data_from_names('./ign_new/images/validation', image_names, shape=(256, 256))
    y = load_data_from_names('./ign_new/annotations/validation', mask_names, shape=(256, 256), mask=True)
    model = ks.models.load_model('./saved_models/semantic_segmentation_new_data_test.h5', custom_objects={'dice_coef_multilabel': dice_coef_multilabel})
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[dice_coef_multilabel], run_eagerly=True)

    predictions = model.predict(x)
    print('Mean dice coefficient multi-label', np_dice_coef_multilabel(y, predictions))

    # for image, pred, y_true in zip(x, predictions, y):
    #
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    #     ax[0].imshow(image)
    #
    #     ax[1].imshow(np.argmax(y_true, axis=-1))
    #
    #     ax[2].imshow(np.argmax(pred, axis=-1))
    #
    #     plt.show()
    # # plt.close()

# RESULTS:

# semantic_segmentation_complete.h5
#       20 epochs,
#       0.1 validation split,
#       learning rate  0.001,
#       elu activation function,
#       8 batch size,
#       256x256 resized images (no augmentation)
#       mean dice coef (test): 0.28 (more or less)

# semantic_segmentation_analyze_me.h5
#       168 epochs,
#       0.1 validationsplit,
#       learning rate 0.001,
#       elu activation function,
#       8 batch size,
#       256x256 resized images (no augmentation),
#       manually stopped (overfitting not detected by earlystopping)
#       mean dice coef (test): 0.4059314455674234

# semantic_segmentation_new_data_partial.h5
#       56 epochs,
#       0.1 validation split,
#       learning rate 0.001,
#       elu activation function,
#       8 batch size,
#       256x256 resized images (no augmentation),
#       manually stopped (learning rate  too low)
#       mean dice coef (test): 0.37742858777140015
