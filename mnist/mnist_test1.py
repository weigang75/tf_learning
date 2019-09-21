# -- coding:utf-8 --
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import PIL.Image as Image
# Just disables the warning, doesn't enable AVX/FMA
import os
import time

# 默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
# 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)


# 只显示 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def auto_norm(data):  # 传入一个矩阵
    mins = data.min(0)  # 返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)  # 返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins  # 最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))  # 生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]  # 返回 data矩阵的行数
    normData = data - np.tile(mins, (row, 1))  # data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges, (row, 1))  # data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData


def show_array(a, title=''):
    # 获取矩阵中最大值和最小值
    _max, _min = find_max_min_value(a)
    if _max == _min:
        plt.imshow(np.uint8(a))
    else:  # 如果最大值和最小值不相等，则需要归一化（0-1之间），并且*255(0-255之间)
        plt.imshow(auto_norm(a) * 255)
    plt.title(title, fontproperties=font_set)
    plt.show()


# 显示图片
def show_bmp(im_arr):
    # 参考：http://blog.csdn.net/u010194274/article/details/50817999
    im = np.array(im_arr)
    im = im.reshape(28, 28)

    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)

    plt.imshow(im, cmap='gray')
    plt.show()


def find_martrix_min_value(data_matrix):
    ''''' 
    功能：找到矩阵最小值 
    '''
    new_data = []
    for i in range(len(data_matrix)):
        new_data.append(min(data_matrix[i]))
    return min(new_data)


def find_martrix_max_value(data_matrix):
    ''''' 
    功能：找到矩阵最大值 
    '''
    new_data = []
    for i in range(len(data_matrix)):
        new_data.append(max(data_matrix[i]))
    return max(new_data)


def find_max_min_value(data_matrix):
    ''''' 
    功能：找到矩阵最大值 
    '''
    new_data = []
    for i in range(len(data_matrix)):
        new_data.append(max(data_matrix[i]))
    return max(new_data), min(new_data)


def show_loge(val):
    plt.figure(1)
    plt.plot(val, np.log(val), 'b--')
    plt.show()


# print("begin")


# 为了用于这个教程，我们使标签数据是"one-hot vectors"。 
# 一个 one_hot 向量除了某一位的数字是1以外其余各维度数字都是0。
# one_hot 标签则是顾名思义，一个长度为n的数组，只有一个元素是1.0，其他元素是0.0。
# 所以在此教程中，数字n将表示成一个只有在第n维度（从0开始）数字为1的10维向量。比如，标签0将表示成([1,0,0,0,0,0,0,0,0,0])。
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 记录开始时间
start_time = time.time()

# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
# 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）
with tf.name_scope('inputs'):
    input_images = tf.compat.v1.placeholder("float", [None, 784], name='input_images')
    # 为了后面计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值
    input_labels = tf.compat.v1.placeholder("float", [None, 10], name='input_labels')

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        # 在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。
        W = tf.Variable(tf.zeros([784, 10]), name='W')
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
    with tf.name_scope('Wx_plus_b'):
        # 首先，我们用tf.matmul(​​X，W)表示x乘以W，对应之前等式里面的，这里x是一个2维张量拥有多个输入。然后再加上b，把和输入到tf.nn.softmax函数里面。
        # 至此，我们先用了几行简短的代码来设置变量，然后只用了一行代码来定义我们的模型。
        # TensorFlow不仅仅可以使softmax回归模型计算变得特别简单，它也用这种非常灵活的方式来描述其他各种数值计算，从机器学习模型对物理学模拟仿真模型。
        # 一旦被定义好之后，我们的模型就可以在不同的设备上运行：计算机的CPU，GPU，甚至是手机！
        result = tf.nn.softmax(tf.matmul(input_images, W) + b, name='result')

with tf.name_scope('loss'):
    # 注意，tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了。我们计算的交叉熵是指整个minibatch的。
    # 计算交叉熵（越小越好 ，注意 cross_entropy 等于负数，所以 tf.reduce_sum(y_ * tf.math.log(y)) 越大越好）: 
    # 交叉熵可在神经网络(机器学习)中作为损失函数，p表示真实标记的分布，q则为训练后的模型的预测标记分布，交叉熵损失函数可以衡量p与q的相似性。
    # 交叉熵作为损失函数还有一个好处是使用sigmoid函数在梯度下降时能避免均方误差损失函数学习速率降低的问题，因为学习速率可以被输出的误差所控制。
    cross_entropy = -tf.reduce_sum(input_labels * tf.math.log(result), name='cross_entropy')
    # 要想明白交叉熵(Cross Entropy)的意义，可以从熵(Entropy) -> KL散度(Kullback-Leibler Divergence) -> 交叉熵这个顺序入手。
    # 当然，也有多种解释方法[1]。
    # 先给出一个“接地气但不严谨”的概念表述：
    #  - 熵：可以表示一个事件A的自信息量，也就是A包含多少信息。
    #  - KL散度：可以用来表示从事件A的角度来看，事件B有多大不同。
    #  - 交叉熵：可以用来表示从事件A的角度来看，如何描述事件B。
    # print(cross_entropy)

with tf.name_scope('train'):
    # 学习时，使用梯度下降优化器算法，要求 cross_entropy（交叉熵）值最小
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 变量需要通过seesion初始化后，才能在session中使用。
# 这一初始化步骤为，为初始值指定具体值（本例当中是全为零），并将其分配给每个变量,可以一次性为所有变量完成此操作
init = tf.compat.v1.global_variables_initializer()

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# 开始初始化变量
sess.run(init)

writer = tf.compat.v1.summary.FileWriter("../logs", sess.graph)

# 用于将 one-hot向量 变为数字（可以忽略）
mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
range_num = 10000
for i in range(range_num):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # if i % 2000 == 0:  # 为了防止打印图片影响性能，则每隔2000个，打印一下图片

    if i % (range_num / 4) == 0 or i >= range_num - 1 or i < 4:
        # 显示标签的 one-hot 向量 及其 数字
        print('[', i, ']->', batch_ys[0], '; 数字=', np.matmul(batch_ys[0], mask).astype(np.uint))
        sta_b = sess.run(b)
        sta_W = sess.run(W)
        sta_W_ = sta_W[300:310, 4:6]  # 因为矩阵太大，截取一部分看看
        print('[', i, ']->', "学习状态b=", sta_b, '\nW=', sta_W_)
        b_ = np.expand_dims(sta_b.T, axis=1)  # np.expand_dims(a) numpy的升维, np.squeeze(a) numpy的降维
        show_array(b_, 'b 矩阵图')
        # t = np.reshape(sta_W.T, (70, 112))  # sta_W.T = (10, 784) -> (70, 112)
        # show_array(np.reshape(sta_W.T, (70, 112)))
        # t = np.reshape(sta_W.T, (140, 56))
        # show_array(np.reshape(sta_W.T, (140, 56)))
        show_array(np.reshape(sta_W.T, (280, 28)), 'W 矩阵图')
        # t = sta_W.T  # (10, 784)
        # print(np.shape(t))
        # show_array(t)
        # show_array(np.reshape(sta_W.T, (112, 70)))
        # 显示数字的图片
        show_bmp(batch_xs[0])
        # 输入 feed_dict 数据，进行学习
    sess.run(train_step, feed_dict={input_images: batch_xs, input_labels: batch_ys})

# argmax 返回的是最大数的索引.argmax 有一个参数axis=1,表示第1维的最大值.
correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(input_labels, 1), name='correct_prediction')

# print('result: ', result)
# print('input_labels: ', input_labels)

# 这里返回一个布尔数组。
# 为了计算我们分类的准确率，我们将布尔值转换为浮点数来代表对、错，然后取平均值。
# 例如：[True, False, True, True]变为[1,0,1,1]，计算出平均值为0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

# 最后，我们计算所学习到的模型在测试数据集上面的正确率。
print('正确率', sess.run(accuracy, feed_dict={input_images: mnist.test.images, input_labels: mnist.test.labels}))

sess.close()
elapse_time = time.time() - start_time
print('耗时（秒）：', elapse_time)
