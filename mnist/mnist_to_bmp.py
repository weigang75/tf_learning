import struct
import numpy as np
import matplotlib.pyplot as plt
# import Image
import PIL.Image as Image

# 参考：http://blog.csdn.net/u010194274/article/details/50817999
# 二进制的形式读入
filename = '/root/PycharmProjects/test/MNIST_data/train-images-idx3-ubyte'
# filename = '/root/PycharmProjects/test/MNIST_data/t10k-images-idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()
# 大端法读入4个unsigned int32
# struct用法参见网站 http://www.cnblogs.com/gala/archive/2011/09/22/2184801.html

index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')
# 将每张图片按照格式存储到对应位置
for image in range(0, numImages):
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    # 这里注意 Image对象的dtype是uint8，需要转换
    im = np.array(im, dtype='uint8')
    im = im.reshape(28, 28)
    # fig=plt.figure()
    # plotwindow=fig.add_subplot(111)
    # plt.imshow(im,cmap='gray')
    # plt.show()
    im = Image.fromarray(im)
    im.save('/root/PycharmProjects/test/MNIST_data/images/train/train_%s.bmp' % image, 'bmp')
    # im.save('/root/PycharmProjects/test/MNIST_data/images/test/test_%s.bmp' % image, 'bmp')
