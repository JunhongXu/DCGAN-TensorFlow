from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from src.DCGAN import DCGAN

mnist = input_data.read_data_sets("MNIST_data", one_hot=False, reshape=False)
N, H, W, C = mnist.train._images.shape
data = np.tanh(mnist.train._images)
data = np.lib.pad(data, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant", constant_values=(0, 0))
plt.imshow(data[0].reshape(32, 32), cmap="gray")
plt.show()
with tf.Session() as sess:
    dcgan = DCGAN(sess, input_dim=(32, 32, 1), g_init=32, d_init=8, name="mnist", lr=0.0001, stddv=0.01)
    dcgan.train(data, 50, test_every=800)
