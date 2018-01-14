#!/usr/bin/python
# encoding=utf-8
# -*- coding:utf-8 -*

import tensorflow as tf
import numpy as np
# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------------------------------------------------
# matmul 演示
# x = [0,0,0,0,0,0,0,1,0,0]
# y = [0,1,2,3,4,5,6,7,8,9]
#
# z = np.matmul(y,x)
#
# print(z)

# ----------------------------------------------------------------------
# argmax 演示

# x = [[5, 2],
#      [6, 4]]
# # -> 6(1),4(1)
#
# y = [[8, 2],
#      [6, 4]]
# # -> 8(0),4(1)
#
# # 最大值的索引值
# z = np.argmax(x, 0)
#
# print(z)
# # print(np.argmax(y, 0))

# ----------------------------------------------------------------------
# reduce_sum 演示
'''
reduce_sum(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None
)
'''
x = [[1, 1, 1],
     [1, 1, 1]]
with tf.Session() as sess:
    print(sess.run(tf.reduce_sum(x)))  # ==> 6
    print(sess.run(tf.reduce_sum(x, 0)))  # ==> [2, 2, 2]
    print(sess.run(tf.reduce_sum(x, 1)))  # ==> [3, 3]
    print(sess.run(tf.reduce_sum(x, 1, keep_dims=True)))  # ==> [[3], [3]]
    print(sess.run(tf.reduce_sum(x, [0, 1])))  # ==> 6
