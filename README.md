# DCGNA Reimplementation in TensorFLow

Link of the original paper: https://arxiv.org/abs/1511.06434

![gan](misc/gen-architecture.png)

#### Code

1. src/DCGAN.py is the main implementation

2. src/ops.py consists of different wrapped operations for TensorFlow

3. main.py starts the training on MNIST dataset
(for now it only trains on MNIST.
Other dataset will be implemented in the future)

#### MINIST Results

Generator's Loss:

![g_loss](misc/g_loss.png)


Discriminator's Loss:

![d_loss](misc/d_loss.png)


Generated Numbers:

![mnist_generated](misc/generated_mnist.png)


#### CelebA Dataset

Generator's Loss:

![celeba_g_loss](misc/celeb_d_loss.png)

Discriminator's loss:

![celeba_d_loss](misc/celeb_d_loss.png)


CelebA Dataset(downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):

![celeba_data](misc/celebA_dataset.png)


Generated Faces

![fake_celeba](misc/celebA_test.png)

#### Reference

https://github.com/carpedm20/DCGAN-tensorflow

https://github.com/yihui-he/GAN-MNIST/blob/master/README.md
