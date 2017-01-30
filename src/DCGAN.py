import tensorflow as tf
from src.ops import *
import numpy as np
from scipy.misc import imsave


class DCGAN(object):
    """
    An implementation of DCGAN
    """
    def __init__(self, sess, input_dim=(64, 64, 3), z_dim=100, lr=0.0002, batch_size=128, g_init=128, d_init=64):

        # model parameters
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.lr = lr
        self.g_init = g_init
        self.d_init = d_init
        self.sess = sess
        self.batch_size= batch_size

        # input to D
        self.x = tf.placeholder(dtype=tf.float32, shape=(batch_size, ) + input_dim, name="x")

        # input to G
        self.z = tf.placeholder(dtype=tf.float32, shape=(batch_size, z_dim), name="z")

        # the output of the generator
        self.G = self.generator(self.z, reuse=None, is_train=True)

        # the probability of input is from real data
        self.D1 = self.discriminator(self.x, reuse=None, is_train=True)
        # the probability of input is from fake data
        self.D2 = self.discriminator(self.G, reuse=True, is_train=True)

        # generator at inference time
        self.G_inference = self.generator(self.z, reuse=True, is_train=False)

        self.g_params = [param for param in tf.trainable_variables() if 'generator' in param.name]
        self.d_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]

        # build the whole model and loss
        self.fake_data_loss, self.real_data_loss, self.generator_loss, self.d_optimizer, self.g_optimizer = self.build()

        # summary writer
        self.writer = tf.summary.FileWriter(logdir="log", graph=sess.graph)
        # summary of generator
        self.g_summary = tf.summary.scalar(name="g/loss", tensor=self.generator_loss)
        # summary of discriminator
        d_loss_summary = tf.summary.scalar("d/loss", tensor=self.fake_data_loss+self.real_data_loss)
        d_loss_fake_summary = tf.summary.scalar("d/loss_fake", tensor=self.fake_data_loss)
        d_loss_real_summary = tf.summary.scalar("d/loss_real", tensor=self.real_data_loss)
        self.d_summary = tf.summary.merge([d_loss_summary, d_loss_fake_summary, d_loss_real_summary])

        self.sess.run(tf.global_variables_initializer())

    def discriminator(self, inpt, reuse, is_train):
        """
        Build D for training or testing. If reuse if True, the input should be the output of generator
        """
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            net = conv2d(x=inpt, num_kernels=self.d_init, name="conv1", activation=lkrelu, padding="SAME",
                         alpha=0.02, is_train=is_train)
            net = conv2d(x=net, num_kernels=self.d_init*2, name="conv2", activation=lkrelu, padding="SAME",
                         alpha=0.02, is_train=is_train)
            net = conv2d(x=net, num_kernels=self.d_init*4, name="conv3", activation=lkrelu, padding="SAME",
                         alpha=0.02, is_train=is_train)
            net = conv2d(x=net, num_kernels=self.d_init*8, name="conv4", activation=lkrelu, padding="SAME",
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
            net = dense_layer(inpt, self.g_init*h**2/32, activation=tf.nn.relu,
                              name="dense_input", use_bn=True, is_train=is_train)
            # reshape the output of dense layer to be H/16, W/16, K*8
            net = tf.reshape(net, (-1, h//16, w//16, self.g_init*8))
            net = transpose_conv2d(net, (self.batch_size, h//8, w//8, self.g_init*4), name="trans_conv1",
                                   is_train=is_train, padding="SAME")
            net = transpose_conv2d(net, (self.batch_size, h//4, w//4, self.g_init*2), is_train=is_train,
                                   padding="SAME")
            net = transpose_conv2d(net, (self.batch_size, h//2, w//2, self.g_init), name="trans_conv3",
                                   is_train=is_train, padding="SAME")
            net = transpose_conv2d(net, (self.batch_size, h, w, c), name="trans_conv4", is_train=is_train,
                                   activation=tf.nn.tanh, padding="SAME")
        return net

    def build(self):
        with tf.variable_scope("discriminator_optimizer"):
            d_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
            # loss for real data
            real_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1, tf.ones_like(self.D1)))
            # loss for fake data
            fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2, tf.zeros_like(self.D2)))
            d_loss = fake_data_loss + real_data_loss
            d_optimize = d_optimizer.minimize(d_loss, var_list=self.d_params)

        with tf.variable_scope("generator_optimizer"):
            g_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2, tf.ones_like(self.D2)))
            g_optimize = g_optimizer.minimize(g_loss, var_list=self.g_params)
        return fake_data_loss, real_data_loss, g_loss, d_optimize, g_optimize

    def train(self, data, max_epoch, test_every=100):
        N, H, W, C = data.shape
        iter_per_epoch = N//self.batch_size
        max_iter = iter_per_epoch * max_epoch

        for step in range(0, max_iter):
            # sample from real data
            random_index = np.random.randint(0, N, self.batch_size)
            x = data[random_index]
            # sample from uniform distribution z
            z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            # train discriminator
            d1_loss, d2_loss, d_summary, _ = self.sess.run([self.real_data_loss, self.fake_data_loss, self.d_summary,
                                                            self.d_optimizer], feed_dict={self.x: x, self.z:z})
            self.writer.add_summary(d_summary, global_step=step)

            # train generator
            g_loss, g_summary, _ = self.sess.run([self.generator_loss, self.g_summary, self.g_optimizer],
                                                 feed_dict={self.z: z})
            self.writer.add_summary(g_summary, global_step=step)

            if step % 10 == 0:
                print("G loss is %s, fake data loss is %s, real data loss is %s, D loss is %s"
                      % (g_loss, d2_loss, d1_loss, d2_loss+d1_loss))

            if step % test_every == 0:
                z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
                # test and save image
                fake_image = self.sess.run(self.G_inference, feed_dict={self.z: z})
                fake_image = np.reshape(fake_image, (self.batch_size, H, W, C))
                imsave("image/%s.png" % step, fake_image[0].reshape(H, W))
                print("Image Saved")
