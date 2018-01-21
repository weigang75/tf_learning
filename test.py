#!/usr/bin/python
# encoding=utf-8
# -*- coding:utf-8 -*

# ImportError: No module named input_data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------------------------------------------------
# # log reduce_sum
# y = tf.constant(
#     [np.power(np.e, 2.0), np.power(np.e, 1.3)])  # tf.random_normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float32)
# y_ = tf.constant(
#     [np.power(np.e, 2.3), np.power(np.e, 1.0)])  # tf.random_normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float32)
#
# # print('np.log(np.e)=', np.log(np.e*np.e))
# print(np.power(10, 2))
#
# # y_ = y
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print('y => ', sess.run(y))
#     print('y_ => ', sess.run(y_))
#     z = tf.log(y)
#     print('tf.log(y) => \n', sess.run(z))
#     cross_entropy = -tf.reduce_sum(y_ * z)
#     print('cross_entropy => \n', sess.run(cross_entropy))

# ----------------------------------------------------------------------
# mnist = input_data.read_data_sets('/root/PycharmProjects/test/MNIST_data', one_hot=True)
# batch_xs, batch_ys = mnist.train.next_batch(100)
# labels = batch_ys  # tf.random_normal([2, 13],dtype=tf.float32)
# with tf.Session() as sess:
#     batch_size = tf.size(labels)
#     print(sess.run(batch_size))
#     labels = tf.expand_dims(labels, 1)
#     indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
#     concated = tf.concat(axis=1, values=[indices, labels])
#     onehot_labels = tf.sparse_to_dense(
#         concated, tf.pack([batch_size, 10]), 1.0, 0.0)
#
#     print(sess.run(onehot_labels))
def fun():
    return 'hello '

def fun1():
    return [fun(),2,3]
a,b,c = fun1()

print(a)
print(b)

# ----------------------------------------------------------------------
# matmul 演示
# x = [0,0,0,0,0,0,0,1,0,0]
# y = [0,1,2,3,4,5,6,7,8,9]
#
# z = np.matmul(y,x)
#
# print(z)

# ----------------------------------------------------------------------
# # argmax 演示
# #                  [arg = 1]
# x = [[5, 2],  # -> 5(0)
#      [6, 4]]  # -> 6(0)
# # -> 6(1),4(1) [arg = 0]
#
# y = [[8, 2],
#      [6, 4]]
# # -> 8(0),4(1)
#
# # 最大值的索引值
# print(np.argmax(x, 0))
# print(np.argmax(x, 1))
# # print(np.argmax(y, 0))

# ----------------------------------------------------------------------
# # reduce_sum 演示
# '''
# reduce_sum(
#     input_tensor,
#     axis=None,
#     keep_dims=False,
#     name=None,
#     reduction_indices=None
# )
# '''
# x = [[1, 1, 1],
#      [1, 1, 1]]
# with tf.Session() as sess:
#     print(sess.run(tf.reduce_sum(x)))  # ==> 6
#     print(sess.run(tf.reduce_sum(x, 0)))  # ==> [2, 2, 2]
#     print(sess.run(tf.reduce_sum(x, 1)))  # ==> [3, 3]
#     print(sess.run(tf.reduce_sum(x, 1, keep_dims=True)))  # ==> [[3], [3]]
#     print(sess.run(tf.reduce_sum(x, [0, 1])))  # ==> 6
