# import numpy as np
import tensorflow as tf

""" Dice loss """

smooth = 1e-5


def dice_coeff(y_true, y_pred):
    dice_coefficient = (2 * tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return dice_coefficient


def dice_loss(y_true, y_pred):
    dice_loss_value = 1 - dice_coeff(y_true, y_pred)
    return dice_loss_value


""" Jaccard Loss """


def jaccard_index(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    jaccard_index = (intersection + smooth) / (union - intersection + smooth)
    return jaccard_index


def jaccard_loss(y_true, y_pred):
    loss = 1 - jaccard_index(y_true, y_pred)
    return loss


""" Binary Cross entropy loss"""


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    loss = -((y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)))
    return tf.reduce_mean(loss)
