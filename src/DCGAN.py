import tensorflow as tf


class DCGAN(object):
    """
    An implementation of DCGAN
    """
    def __init__(self, input_dim=(64, 64, 3), z_dim=100, batch_size=64, lr=1e-4):
        """
        Parameters:
            input_dim: a tuple of (H, W, C) of input image shape to the discriminator
                or output image shape of the generator
            z_dim: generator input dimension
            batch_size: int
            lr: learning rate
        """

        # model parameters
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lr = lr

        # input to D
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, ) + input_dim, name="x")

        # input to G
        self.z = tf.placeholder(dtype=tf.float32, shape=(None, z_dim), name="z")

        # the output of the generator
        self.G = self.generator()

        # the probability of input is from real data
        self.D1 = self.discriminator(self.x, reuse=None)
        # the probability of input is from fake data
        self.D2 = self.discriminator(self.z, reuse=True)

        # build the whole model and loss
        self.fake_data_loss, self.real_data_loss, self.generator_loss, self.d_optimizer, self.g_optimizer = self.build()

    def discriminator(self, inpt, reuse):
        """
        Build D. If reuse if True, the input should be the output of generator
        """
        with tf.variable_scope("discriminator"):
            net = tf.nn.conv2d(input=inpt,)
            pass