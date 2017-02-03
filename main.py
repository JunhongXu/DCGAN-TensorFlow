from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from src.DCGAN import DCGAN
from src.utilities import recover_img
from src.utilities import extract_data

dataset_name = "celeb"
H, W = 64, 64


if __name__ == '__main__':
    if dataset_name == "mnist":
        data = extract_data()
        C = 1
    elif dataset_name == "celeb":
        data = extract_data("img_align_celeba", H=H, W=W)
        C = 3
    index = np.random.randint(0, data.shape[0])
    plt.imshow(recover_img(data[index].reshape([H, W, C])))
    plt.show()

    with tf.Session() as sess:
        dcgan = DCGAN(sess, input_dim=(H, W, C), name=dataset_name, g_init=128, d_init=128)
        dcgan.train(data, 200, test_every=200)
