"""
    This module is for extracting the data, padding or cropping the data,
    pre-processing the data, and save evaluation images
"""
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt


def extract_data(dataset_name='mnist', H=32, W=32):
    """
        Extract the data given the name of the dataset.
        If the height or width is smaller than H or W, pad the data with 0's on the two axis.
        If the height or width is larger than H or W, crop the data on the two axis.
    """

    if dataset_name == "mnist":
        mnist = input_data.read_data_sets("MNIST_data", one_hot=False, reshape=False)
        data = mnist.train._images
        n, h, w, c = data.shape
        # data = np.tanh(mnist.train._images)
    else:
        # data =
        pass

    # pad or crop
    h_diff = h - H
    w_diff = w - W
    if h_diff < 0:
        h_diff = abs(h_diff)
        data = np.pad(data, ((0, 0), (h_diff//2, h_diff//2), (0, 0), (0, 0)), mode="constant", constant_values=(0, 0))
    else:
        data = data[:, h_diff//2:h-h_diff//2, :, :]

    if w_diff < 0:
        w_diff = abs(w_diff)
        data = np.pad(data, ((0, 0), (0, 0), (w_diff//2, w_diff//2), (0, 0)), mode="constant", constant_values=(0, 0))
    else:
        data = data[:, :, w_diff//2:w-w_diff//2, :]

    return np.tanh(data)
