import tensorflow as tf
from layers import dense


class Discriminator:
    def __init__(self, is_training):
        self.name = 'model/discriminator'
        self.is_training = is_training

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = dense(x, 500, activation_='lrelu')
            _x = dense(_x, 500, activation_='lrelu')
            _x = dense(_x, 1, activation_=None)
            return _x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
