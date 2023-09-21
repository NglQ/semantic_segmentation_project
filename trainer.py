# Utilities
import os
import datetime

import signal
import sys

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
from nn_builder import build_cnn, get_model, mobileunet, mod_mobileunet
from callbacks import DisplayCallback
from custom_metrics import dice_coef_multilabel
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

CNN_model = None


def signal_handler(sig, frame):
    if CNN_model is not None:
        CNN_model.save('./saved_models/semantic_segmentation_panic_new_data.h5')
        print('Model saved', flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def train_model(pretrained_model=None):  # pretrained_model is the path to the pretrained model to be used
    image_names = os.listdir('./ign_new/images/training')[:1000]
    mask_names = os.listdir('./ign_new/annotations/training')[:1000]

    image_names.sort()
    mask_names.sort()

    # WARNING This should contain a better sample of the images (not just the first 1000)
    x = load_data_from_names('./ign_new/images/training', image_names, shape=(256, 256))
    y = load_data_from_names('./ign_new/annotations/training', mask_names, shape=(256, 256), mask=True)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

    global CNN_model
    if pretrained_model:
        CNN_model = ks.models.load_model(pretrained_model, custom_objects={'dice_coef_multilabel': dice_coef_multilabel})
        CNN_model.compile(optimizer=Adam(learning_rate=1e-3),
                          loss='binary_crossentropy',
                          metrics=[dice_coef_multilabel],
                          run_eagerly=True)
    else:
        CNN_model = mod_mobileunet(input_size=(256, 256, 3))

    print(CNN_model.summary())

    # Set hyperparameters
    BATCH_SIZE = 8
    N_EPOCHS = 300

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    cBack = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef_multilabel', patience=15, mode='max', restore_best_weights=True)
    checkPointCb = tf.keras.callbacks.ModelCheckpoint(filepath="./saved_models/semantic_segmentation_new_data.h5",
                                                      monitor='val_dice_coef_multilabel', mode='max', save_best_only=True)

    display_callback = DisplayCallback(epoch_interval=1, test_images=x_val, test_masks=y_val, BATCH_SIZE=BATCH_SIZE)
    # dice_metric_callback = Metrics(7, CNN_model)

    # Training
    hist = CNN_model.fit(x_train, y_train,
                         batch_size=BATCH_SIZE,
                         epochs=N_EPOCHS,
                         validation_data=(x_val, y_val),
                         callbacks=[tensorboard_callback, cBack, checkPointCb])  # , display_callback])

    CNN_model.save('./saved_models/semantic_segmentation_new_data.h5')

    return CNN_model, hist




CNN_model = None

def signal_handler(sig, frame):
    if CNN_model is not None:
        CNN_model.save('/content/drive/MyDrive/saved_models/semantic_segmentation_panic_new_data.h5')
        print('Model saved', flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def train_model(pretrained_model=None):  # pretrained_model is the path to the pretrained model to be used
    image_names = os.listdir('/content/drive/MyDrive/ign_new/images/training')[:1000]
    mask_names = os.listdir('/content/drive/MyDrive/ign_new/annotations/training')[:1000]

    image_names.sort()
    mask_names.sort()

    x = load_data_from_names('/content/drive/MyDrive/ign_new/images/training', image_names, shape=(256, 256))
    y = load_data_from_names('/content/drive/MyDrive/ign_new/annotations/training', mask_names, shape=(256, 256), mask=True)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

    global CNN_model
    if pretrained_model:
        CNN_model = ks.models.load_model(pretrained_model, custom_objects={'dice_coef_multilabel': dice_coef_multilabel})
        CNN_model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[dice_coef_multilabel], run_eagerly=True)
    else:
        CNN_model = mod_mobileunet(input_size=(256, 256, 3))

    print(CNN_model.summary())

    # Set hyperparameters
    BATCH_SIZE = 8
    N_EPOCHS = 300

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    cBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkPointCb = tf.keras.callbacks.ModelCheckpoint(filepath="/content/drive/MyDrive/saved_models/semantic_segmentation_new_data.h5",
                                                      monitor='val_dice_coef_multilabel', mode='max', save_best_only=True)

    display_callback = DisplayCallback(epoch_interval=1, test_images=x_val, test_masks=y_val, BATCH_SIZE=BATCH_SIZE)
    # dice_metric_callback = Metrics(7, CNN_model)

    # Training
    hist = CNN_model.fit(x_train, y_train,
                         batch_size=BATCH_SIZE,
                         epochs=N_EPOCHS,
                         validation_data=(x_val, y_val),
                         callbacks=[tensorboard_callback, cBack, checkPointCb])  # , display_callback])

    CNN_model.save('/content/drive/MyDrive/saved_models/semantic_segmentation_new_data.h5')

    return CNN_model, hist



