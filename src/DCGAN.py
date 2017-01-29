import tensorflow as tf
from src.ops import *

class DCGAN(object):
    """
    An implementation of DCGAN
    """
    def __init__(self, input_dim=(64, 64, 3), z_dim=100, batch_size=64, lr=1e-4, init_num_kernels=64):
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
        self.init_num_kernels = init_num_kernels

        # input to D
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, ) + input_dim, name="x")

        # input to G
        self.z = tf.placeholder(dtype=tf.float32, shape=(None, z_dim), name="z")

        # the output of the generator
        self.G = self.generator(self.z, reuse=None, is_train=True)

        # the probability of input is from real data
        self.D1 = self.discriminator(self.x, reuse=None, is_train=True)
        # the probability of input is from fake data
        self.D2 = self.discriminator(self.G, reuse=True, is_train=True)

        self.g_params = [param for param in tf.trainable_variables() if 'generator' in param]
        self.d_params = [param for param in tf.trainable_variables() if 'discriminator' in param]

        # build the whole model and loss
        self.fake_data_loss, self.real_data_loss, self.generator_loss, self.d_optimizer, self.g_optimizer = self.build()

    def discriminator(self, inpt, reuse, is_train):
        """
        Build D for training or testing. If reuse if True, the input should be the output of generator
        """
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            net = conv2d(x=inpt, num_kernels=self.init_num_kernels, name="conv1", activation=lkrelu, padding="SAME",
                         alpha=0.02, is_train=is_train)
            net = conv2d(x=net, num_kernels=self.init_num_kernels*2, name="conv2", activation=lkrelu, padding="SAME",
                         alpha=0.02, is_train=is_train)
            net = conv2d(x=net, num_kernels=self.init_num_kernels*4, name="conv3", activation=lkrelu, padding="SAME",
                         alpha=0.02, is_train=is_train)
            net = conv2d(x=net, num_kernels=self.init_num_kernels*8, name="conv4", activation=lkrelu, padding="SAME",
                         alpha=0.02, is_train=is_train)
            net = dense_layer(x=net, num_neurons=1, name="output", activation=tf.identity, is_train=is_train)
        return net

    def generator(self, inpt, reuse, is_train):
        """
        Build generator G for training or testing
        """
        h, w, c = self.input_dim
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            net = dense_layer(inpt, self.init_num_kernels*h**2/32, activation=tf.nn.relu,
                              name="dense_input", use_bn=True, is_train=is_train)
            # reshape the output of dense layer to be H/16, W/16, K*8
            net = tf.reshape(net, (-1, h//16, w//16, self.init_num_kernels*8))
            net = transpose_conv2d(net, self.init_num_kernels*4, h//8, w//8, name="trans_conv1", is_train=is_train)
            net = transpose_conv2d(net, self.init_num_kernels*2, h//4, w//4, name="trans_conv2", is_train=is_train)
            net = transpose_conv2d(net, self.init_num_kernels, h//2, w//2, name="trans_conv3", is_train=is_train)
            net = transpose_conv2d(net, c, h, w, name="trans_conv4", is_train=is_train, activation=tf.nn.tanh)
        return net

    def build(self):
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        with tf.variable_scope("discriminator_optimizer"):
            d_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # loss for real data
            real_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1, tf.ones_like(self.D1)))
            # loss for fake data
            fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2, tf.zeros_like(self.D2)))
            d_loss = fake_data_loss + real_data_loss
            d_optimize = d_optimizer.minimize(d_loss, var_list=self.d_params)

        with tf.variable_scope("generator_optimizer"):
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.D2, tf.ones_like(self.D2))
            g_optimize = g_optimizer.minimize(g_loss, var_list=self.g_params)
        return fake_data_loss, real_data_loss, g_loss, d_optimize, g_optimize


if __name__ == '__main__':
    DCGAN()