from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from src.DCGAN import DCGAN
from src.utilities import recover_img
from src.utilities import extract_data
import argparse
import os

parser = argparse.ArgumentParser(description="Run the DCGAN")

# dataset
parser.add_argument("dataset", type=str, choices=["mnist", "celeb"])
parser.add_argument("--H", type=int, default=64)
parser.add_argument("--W", type=int, default=64)

# model hyperparameters
parser.add_argument("--init_g", type=int, default=128, help="Initial number of kernels in the G")
parser.add_argument("--init_d", type=int, default=128, help="Initial number of kernels in the D")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--stddv", type=float, default=2e-2)
parser.add_argument("--test_every", type=int, default=200)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--restore", action="store_true")
parser.add_argument("--is_train", action="store_true")


if __name__ == '__main__':
    args = parser.parse_args()
    g_init = args.init_g
    d_init = args.init_d
    batch_size= args.batch_size
    lr = args.lr
    stddv = args.stddv
    test_every = args.test_every
    num_epochs = args.num_epochs
    restore = args.restore
    is_train = args.is_train

    dataset_name = args.dataset
    H = args.H
    W = args.W

    if is_train:
        if dataset_name == "mnist":
            data = extract_data()
        else:
            print("Here haha", type(is_train), is_train)
            data = extract_data("img_align_celeba", H=H, W=W)
        C = data.shape[-1]
        index = np.random.randint(0, data.shape[0])
        plt.imshow(recover_img(data[index].reshape([H, W, C])))
        plt.show()
    C = 1 if dataset_name is 'mnist' else 3

    with tf.Session() as sess:
        dcgan = DCGAN(sess, input_dim=(H, W, C), name=dataset_name, g_init=g_init, d_init=d_init, batch_size=batch_size,
                      stddv=stddv, lr=lr, restore=restore)
        if is_train:
            # train the model and store test images into image/dataset_name folder
            dcgan.train(data, num_epochs, test_every=test_every)
        else:
            # evaluate the model and store evaluation images into image/dataset_name/test folder
            dcgan.evaluate()
