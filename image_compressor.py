
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import ipywidgets as widgets

from skimage import io
from sklearn.cluster import KMeans
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider

img_dir ='./images'

def data_preprocessing():
    img = io.imread('images/1-Saint-Basils-Cathedral.jpg')
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(img)
    img_data = (img/255.0).reshape(-1, 3)
    return img_data

def visualizing_color_space(img_data):
    from plot_utils import plot_utils

    x = plot_utils(img_data, title='Input color space: Over 16 million possible colors')
    x.colorSpace()

def color_compression(image=os.listdir(img_dir), k=IntSlider(min=1,max=256, step=1, values=16), 
                        continuous_update=False, layout=dict(width='100%')):
    input_image = io.imread(img_dir+image)
    img_data = (input_image/255.0).reshape(-1, 3)

    kmeans = KMeans(k).fit(img_data)
    k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]
    k_img = np.reshape(k_colors, (input_image.shape))

data_preprocessing()    