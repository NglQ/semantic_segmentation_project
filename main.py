# Utilities
import os
import datetime
from data_loader import load_data_from_names
from glob import glob
import sys

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
from trainer import train_model
from tester import test_model
from dataset_creator import create_dataset
from data_analyzer import analyze_data

# tf.config.experimental_run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

if __name__ == '__main__':
    pretrained_model = './saved_models/semantic_segmentation_new_data_partial.h5'
    train = input('train: 0\ntrain (model): 1\ntest: 2\ncreate dataset: 3\nanalyze data: 4\n')

    if train == '0':
        CNN_model, hist = train_model()
    elif train == '1':
        CNN_model, hist = train_model(pretrained_model)
    elif train == '2':
        test_model()
    elif train == '3':
        create_dataset()
    elif train == '4':
        analyze_data()
    else:
        sys.exit(0)

    # # Check overfit
    # loss_history = hist.history['loss']
    # val_loss_history = hist.history['val_loss']
    #
    # acc_history = hist.history['accuracy']
    # val_acc_history = hist.history['val_accuracy']
    #
    # plt.plot(loss_history)
    # plt.plot(val_loss_history)
    # plt.grid()
    # plt.xlabel('Epoch')
    # plt.legend(['Loss', 'Val Loss'])
    # plt.title('Loss')
    # plt.show()
    #
    # plt.plot(acc_history)
    # plt.plot(val_acc_history)
    # plt.grid()
    # plt.xlabel('Epoch')
    # plt.legend(['Accuracy', 'Val Accuracy'])
    # plt.title('Accuracy')
    # plt.show()

# TODO Find a way to balance the classes in the dataset
# TODO Choose the best images in the dataset
# TODO Train the network with the new dataset ----> Ongoing
# TODO Use dice loss function (try it after getting good results with cross entropy) ----> Pretty risky


