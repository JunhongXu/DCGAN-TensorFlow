from src.utilities import *
import argparse

parser = argparse.ArgumentParser(description="Use this module to download dataset.")
parser.add_argument('name', type=str, help="The name of the dataset[mnist or celebA]")


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.name
    print("Downloading %s dataset" % dataset)
    if dataset == "mnist":
        data = extract_data(dataset)
        print("Shape of MNIST data %s" % data.shape)
    else:
        #TODO: download celebA dataset or other datasets
        # maybe_download()
        pass
