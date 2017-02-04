from src.utilities import *
import argparse


# some download URLs
CELEB_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1'

parser = argparse.ArgumentParser(description="Use this module to download dataset.")
parser.add_argument('name', type=str, choices=["mnist", "celebA"], help="The name of the dataset[mnist/celebA]")


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.name
    print("Downloading %s dataset" % dataset)
    if dataset == "mnist":
        data = extract_data(dataset)
        print("Shape of MNIST data %s" % str(data.shape))
    elif dataset == "celebA":
            maybe_download("data", "%s.zip" % dataset, CELEB_URL)
