import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import ipywidgets as widgets

from skimage import io
from sklearn.cluster import MiniBatchKMeans
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider


def visualizing_color_space(img_data):
    from plot_utils import plot_utils

    x = plot_utils(img_data, title='Input color space: Over 16 million possible colors')
    x.colorSpace()

def color_compression(k, pic_name):
    input_img = io.imread('originals/'+pic_name)
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(input_img)
    img_data = (input_img/255.0).reshape(-1, 3)

    kmeans = MiniBatchKMeans(k).fit(img_data)
    k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]
    k_img = np.reshape(k_colors, (input_img.shape))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('K-means image compression', fontsize=20)

    ax1.set_title('Compressed')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(k_img)
    ax2.set_title('Original (16,777, 216 colors)')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(input_img)

    plt.subplots_adjust(top=0.85)
    plt.show()

    io.imsave('compressed/'+pic_name, k_img)


color_compression(6, '2-new-york-skyline.jpg')