import tensorflow as tf


def conv2d(x, num_kernels, kernel_h=5, kernel_w=5, strides=2, padding="VALID", name="conv2d",
           use_bn=True, activation=tf.nn.relu, is_train=True, stddv=0.02):
    """
    Wrapper function for convolutional layer
    """
    reuse = True if not is_train else None
    n, h, w, c = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable(name="weight", initializer=tf.truncated_normal_initializer(stddev=stddv),
                            shape=(kernel_h, kernel_w, c, num_kernels))
        bias = tf.get_variable(name="bias", initializer=tf.constant_initializer(0.01), shape=num_kernels)
        y = tf.nn.conv2d(x, w, (1, strides, strides, 1), padding, name=name)
        y = tf.nn.bias_add(y, bias)
        y = batch_norm(y, activation, tf.get_variable_scope().name, reuse, is_train)
        return y
    

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


def batch_norm(x, activation, name, reuse, is_train=True):
    name = '%s/%s' % name, 'bn'
    y = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, reuse=reuse,
                                     is_training=is_train, scope=name, activation_fn=activation)
    return y
