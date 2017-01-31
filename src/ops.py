import tensorflow as tf


def conv2d(x, num_kernels, kernel_h=5, kernel_w=5, strides=2, padding="VALID", name="conv2d",
           use_bn=True, activation=tf.nn.relu, alpha=None, is_train=True, stddv=0.02):
    """
    Wrapper function for convolutional layer
    """
    n, h, w, c = x.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable(name="weight", initializer=tf.truncated_normal_initializer(stddev=stddv),
                            shape=(kernel_h, kernel_w, c, num_kernels))
        bias = tf.get_variable(name="bias", initializer=tf.constant_initializer(0.01), shape=num_kernels)
        y = tf.nn.conv2d(x, w, (1, strides, strides, 1), padding)
        y = tf.nn.bias_add(y, bias)

        if use_bn:
            y = batch_norm(y, tf.get_variable_scope().name, is_train)

        print("Convolutional 2D Layer %s, kernel size %s, output size %s Reuse:%s"
              % (tf.get_variable_scope().name, (kernel_h, kernel_w, c, num_kernels), y.get_shape().as_list(),
                 tf.get_variable_scope().reuse))
        if alpha is None:
            y = activation(y)
        else:
            y = activation(y, alpha)
    return y


def transpose_conv2d(x, output_shape, kernel_h=5, kernel_w=5, activation=tf.nn.relu, stride=2, padding="VALID",
                     use_bn=True, is_train=True, stddv=0.02, name="transpose_conv2d"):
    n, h, w, c = x.get_shape().as_list()
    num_kernels = output_shape[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(name="weight", initializer=tf.truncated_normal_initializer(stddev=stddv),
                            shape=(kernel_h, kernel_w, num_kernels, c))
        bias = tf.get_variable(name="bias", initializer=tf.constant_initializer(0.01), shape=num_kernels)
        y = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, padding=padding,
                                   strides=(1, stride, stride, 1))
        y = tf.nn.bias_add(y, bias)
        if use_bn:
            y = batch_norm(y, tf.get_variable_scope().name, is_train)

        print("Transposed Convolutional 2D Layer %s, kernel size %s, output size %s Reuse:%s"
              % (tf.get_variable_scope().name, (kernel_h, kernel_w, c, num_kernels), y.get_shape().as_list(),
                 tf.get_variable_scope().reuse))
    return activation(y)


def dense_layer(x, num_neurons, name, activation, use_bn=False, is_train=True, stddv=0.02):
    if len(x.get_shape().as_list()) > 2:
        n, h, w, c = x.get_shape().as_list()
        d = h * w * c
    else:
        n, d = x.get_shape().as_list()
    with tf.variable_scope(name):
        # flatten x
        x = tf.reshape(x, (-1, d))
        w = tf.get_variable("weight", shape=(d, num_neurons), initializer=tf.random_normal_initializer(stddev=stddv))
        b = tf.get_variable("bias", shape=num_neurons, initializer=tf.constant_initializer(0.01))
        y = tf.matmul(x, w) + b
        if use_bn:
            y = batch_norm(y, name=tf.get_variable_scope().name, is_train=is_train)
        print("Dense Layer %s, output size %s" % (tf.get_variable_scope().name, y.get_shape().as_list()))
    return activation(y)


def lkrelu(x, alpha, name="leaky_relu"):
    """
    An implementation of Leaky Relu
    y = x if x > 0
    y = alpha * x if x < 0

    Parameters:
        x: a tensor
        alpha: int representing the slop
        name: str representing the name of this op
    """
    with tf.name_scope(name):
        y = tf.maximum(x, alpha * x)
    return y


def batch_norm(x, name, is_train=True):
    name = '%s/%s' % (name, 'bn')
    y = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                     is_training=is_train, scope=name)
    print("Batch Normalization Layer %s. Reuse: %s. Is training %s" % (name, tf.get_variable_scope().reuse, is_train))
    return y



