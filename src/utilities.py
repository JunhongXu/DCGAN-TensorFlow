"""
    This module is for extracting the data, padding or cropping the data,
    pre-processing the data, and save evaluation images
"""
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import urllib.request as request
import os
import sys
import zipfile
from scipy.misc import imread
from scipy.misc import imresize


def extract_data(dataset_name='mnist', H=32, W=32, num_images=None):
    """
        Extract the data given the name of the dataset.
        If the height or width is smaller than H or W, pad the data with 0's on the two axis.
        If the height or width is larger than H or W, crop the data on the two axis.
    """

    if dataset_name == "mnist":
        def _pad_or_crop(data):
            # pad or crop
            h_diff = h - H
            w_diff = w - W
            if h_diff < 0:
                # pad
                h_diff = abs(h_diff)
                data = np.pad(data, ((0, 0), (h_diff//2, h_diff//2), (0, 0), (0, 0)), mode="constant", constant_values=(0, 0))
            else:
                # resize
                data = data[:, h_diff//2:h-h_diff//2, :, :]

            if w_diff < 0:
                w_diff = abs(w_diff)
                data = np.pad(data, ((0, 0), (0, 0), (w_diff//2, w_diff//2), (0, 0)), mode="constant", constant_values=(0, 0))
            else:
                data = data[:, :, w_diff//2:w-w_diff//2, :]
            return data
        mnist = input_data.read_data_sets("MNIST_data", one_hot=False, reshape=False)
        data = mnist.train._images
        n, h, w, c = data.shape
        data = np.tanh(mnist.train._images)
        data = _pad_or_crop(data)
    else:
        dataset_name = os.path.join('data', dataset_name)
        dirs = os.listdir(dataset_name)
        print("The number of image files is %s" % len(dirs))
        images = []
        for index, image_file in enumerate(dirs):
            img = imread(os.path.join(dataset_name, image_file))
            # resize the image
            h, w, c = img.shape
            diff = h - w
            if diff < 0:
                img = img[:, abs(diff)//2:w-abs(diff), :]
            else:
                img = img[diff//2:h-abs(diff), :, :]
            img = imresize(img, size=(H, W))
            images.append(img)
            print("Extracting image number %s" % index)
            if num_images:
                if index == num_images:
                    break
        data = np.array(images)/127.5 - 1
        n, h, w, c = data.shape
    return data


def recover_img(image):
    return (image+1.)/2.


def _download_file(url, save_dir):
    # specify how many bytes to read each time
    block_size = 8192
    url = request.urlopen(url)
    f = open(save_dir, 'wb')
    file_len = 0
    total_size = url.headers["Content-Length"]
    print("Total size of the file %s" % url.headers["Content-Length"])
    index = 0
    while True:
        buffer = url.read(block_size)
        file_len += len(buffer)
        if not buffer:
            print("Finish downloading")
            break
        f.write(buffer)
        if index % 100 == 0:
            print("Downloading... %s" %(file_len/int(total_size)))
        index += 1
        sys.stdout.flush()
    f.close()


def maybe_download(path_to_save, file_name, url=None):
    """
        Check if the data file exists. If it does, extract the file. If it does not, download and extracts the file.
    """
    file_path = os.path.join(path_to_save, file_name)

    if not os.path.exists(file_path):
        # download
        os.makedirs(path_to_save)
        _download_file(url, file_path)
    # extract file
    with zipfile.ZipFile(file_path) as zf:
        print("Extracting data...")
        zf.extractall(path_to_save)
        print("Finished...")
    os.remove(file_path)


if __name__ == '__main__':
    # maybe_download('data', 'celeba.zip',
    #               'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1')
    print(extract_data('img_align_celeba', H=64, W=64).shape)

