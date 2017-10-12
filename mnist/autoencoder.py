import tensorflow as tf
from blocks import conv_block
from layers import dense, flatten, reshape


class AutoEncoder:
    def __init__(self, input_shape, latent_dim, last_activation='tanh',
                 color_mode='rgb', normalize='batch', upsampling='deconv', is_training=True):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.last_activation = last_activation
        self.name = 'model/generator'
        assert color_mode in ['grayscale', 'gray', 'rgb']
        self.channel = 1 if color_mode in ['grayscale', 'gray'] else 3
        self.normalize = normalize
        self.upsampling = upsampling
        self.is_training = is_training

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            with tf.variable_scope('Encoder'):
                _x = conv_block(x, filters=16, activation_='lrelu', normalization=self.normalize, sampling='same')
                _x = conv_block(_x, filters=16, activation_='lrelu', normalization=self.normalize, sampling='down')

                _x = conv_block(_x, filters=32, activation_='lrelu', normalization=self.normalize, sampling='same')
                _x = conv_block(_x, filters=32, activation_='lrelu', normalization=self.normalize, sampling='down')

                _x = conv_block(_x, filters=64, activation_='lrelu', normalization=self.normalize, sampling='same')
                _x = conv_block(_x, filters=64, activation_='lrelu', normalization=self.normalize, sampling='down')

                current_shape = _x.get_shape().as_list()[1:]
                _x = flatten(_x)
                _x = dense(_x, 64, activation_='lrelu')
                encoded = dense(_x, self.latent_dim)

            with tf.variable_scope('Decoder'):
                _x = dense(encoded, 64, activation_='lrelu')
                _x = dense(_x, current_shape[0] * current_shape[1] * current_shape[2], activation_='lrelu')
                _x = reshape(_x, current_shape)

                _x = conv_block(_x, filters=64, activation_='lrelu',
                                normalization=self.normalize, sampling=self.upsampling)
                _x = conv_block(_x, filters=32, activation_='lrelu',
                                normalization=self.normalize, sampling='same')

                _x = conv_block(_x, filters=32, activation_='lrelu',
                                normalization=self.normalize, sampling=self.upsampling)
                _x = conv_block(_x, filters=16, activation_='lrelu',
                                normalization=self.normalize, sampling='same')

                _x = conv_block(_x, filters=16, activation_='lrelu',
                                normalization=self.normalize, sampling=self.upsampling)
                _x = conv_block(_x, filters=self.channel, activation_=self.last_activation,
                                normalization=None, sampling='same')

            return encoded, _x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    @property
    def encoder_vars(self):
        return [var for var in tf.global_variables() if self.name in var.name and 'Encoder' in var.name]
