import tensorflow as tf
from tensorflow.contrib.keras.python.keras import layers as kl
from tensorflow.contrib import layers as tl


def dense(x, units, activation_=None):
    return activation(kl.Dense(units, activation=None)(x),
                      activation_)


def conv2d(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation_=None):
    return activation(kl.Conv2D(filters, kernel_size, strides, padding, activation=None)(x),
                      activation_)


def conv2d_transpose(x, filters, kernel_size=(3, 3), strides=(2, 2), padding='same', activation_=None):
    return activation(kl.Conv2DTranspose(filters, kernel_size, strides, padding, activation=None)(x),
                      activation_)


def subpixel_conv2d(x, filters, kernel_size=(3, 3), **kwargs):
    with tf.name_scope(subpixel_conv2d.__name__):
        _x = conv2d(x, filters * 4, kernel_size, strides=(1, 1), activation_=None)
        _x = pixel_shuffle(_x)
    return _x


def pixel_shuffle(x, r=2):
    with tf.name_scope(pixel_shuffle.__name__):
        bs = tf.shape(x)[0]
        _, h, w, c = x.get_shape().as_list()

        _x = tf.transpose(x, (0, 3, 1, 2))
        _x = tf.reshape(_x, (bs, r, r, c // (r ** 2), h, w))
        _x = tf.transpose(_x, (0, 3, 4, 1, 5, 2))
        _x = tf.reshape(_x, (bs, c // (r ** 2), h * r, w * r))
        _x = tf.transpose(_x, (0, 2, 3, 1))
    return _x


def reshape(x, target_shape):
    return kl.Reshape(target_shape)(x)


def activation(x, func=None):
    if func == 'lrelu':
        return kl.LeakyReLU(0.2)(x)
    else:
        return kl.Activation(func)(x)


def batch_norm(x, is_training=True):
    return tl.batch_norm(x, updates_collections=None, is_training=is_training)


def layer_norm(x, is_training=True):
    return tl.layer_norm(x, is_training=is_training)


def flatten(x):
    return kl.Flatten()(x)