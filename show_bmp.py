import struct
import numpy as np
import matplotlib.pyplot as plt
# import Image
import PIL.Image as Image
import show_bmp


def show(im_arr):
    im = np.array(im_arr)
    im = im.reshape(28, 28)

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im, cmap='gray')
    plt.show()
