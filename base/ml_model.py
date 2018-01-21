import tensorflow as tf
import numpy as np


def model_1(in_x, in_w, in_b):
    return tf.multiply(in_w, in_x) + in_b


def model_2(in_x, in_w, in_b):
    return tf.multiply(in_w, in_x) + in_b + 100


def model_3(in_x, in_w, in_b):
    return tf.multiply(tf.add(in_w, 123.0), in_x) + in_b


def model_4(in_x, in_w, in_b):
    return tf.multiply(tf.add(in_w, 123.0), in_x) + in_b + 200.0


selected_model = model_1