import tensorflow as tf
import numpy as np
# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import funs as f
import ml_model as model

# 获取训练数据
def get_train(size):
    train_input = np.random.random(size=size)
    train_output = np.zeros([size, ])
    for i in range(size):
        train_output[i] = f.unknow(train_input[i])
    return train_input, train_output


# 获取测试数据
def get_test(size):
    train_input = np.random.random(size=size)
    train_output = np.zeros([size, ])
    for i in range(size):
        train_output[i] = np.round(f.unknow(train_input[i]), 7)
    return train_input, train_output


# 获取训练数据进行学习
x_data, y_data = get_train(30000)

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([1]))

select_model = model.selected_model

y = select_model(x_data, W, b)

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 拟合平面，开始训练
for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))

result_W = sess.run(W)
result_b = sess.run(b)

# 得出结果
print('W =', result_W, '; b =', result_b)

test_num = 500


# 获取测试数据进行
test_input, test_output = get_test(test_num)
loss = 0.0
for step in range(0, np.size(test_input)):
    test_x = test_input[step]
    test_y = sess.run(select_model(test_x, result_W, result_b))[0]
    loss += np.abs(np.round(test_y - test_output[step], 7))
    if step % 100 == 0:
        print(step, '预测值 :', test_y, '; 实际值 :', test_output[step], '; 误差 : %f' % (loss*100.0/test_num))


print('最后误差 : %f' % (loss*100.0/test_num))
sess.close()
