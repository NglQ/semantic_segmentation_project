import tensorflow.keras as keras
import numpy as np
import tensorflow as tf


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels=7):
    dice = 0
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    for index in range(numLabels):
        dice += dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice/numLabels


def np_dice_coef_multilabel(y_true, y_pred, numLabels=7):
    dice = 0
    for index in range(numLabels):
        dice += dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice/numLabels
