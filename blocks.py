from layers import *


def conv_block(x, filters, activation_, kernel_size=(3, 3), is_training=True,
               sampling='same', normalization=None, dropout=0.0, mode='conv_first'):
    assert mode in ['conv_first', 'normalization_first']
    assert sampling in ['deconv', 'subpixel', 'down', 'same']
    assert normalization in ['batch', 'layer', None]

    conv_func = conv2d_transpose if sampling == 'deconv' \
        else subpixel_conv2d if sampling == 'subpixel'\
        else conv2d
    normalize = batch_norm if normalization == 'batch' \
        else layer_norm if normalization == 'layer' \
        else None
    strides = (1, 1) if sampling in ['same', 'subpixel'] else (2, 2)

    with tf.name_scope(conv_block.__name__):
        if mode == 'conv_first':
            _x = conv_func(x, filters, kernel_size=kernel_size, activation_=None, strides=strides)

            if normalize is not None:
                _x = normalize(_x, is_training)

            _x = activation(_x, activation_)

            if dropout != 0:
                _x = kl.Dropout(dropout)(_x)

        else:
            if normalization is None:
                raise ValueError
            else:
                _x = normalize(x, is_training)

            _x = activation(_x, activation_)
            _x = conv_func(_x, filters, activation_=None, is_training=is_training, strides=strides)
        return _x


def residual_block(x, filters, activation_, kernel_size=(3, 3), is_training=True, sampling='same',
                   normalization=None, dropout=0.0, mode='conv_first'):
    with tf.name_scope(residual_block.__name__):
        _x = conv_block(x, filters, activation_, kernel_size, is_training, sampling, normalization, dropout, mode)
        _x = conv_block(_x, filters, None, kernel_size, is_training, sampling, normalization, dropout, mode)
        return _x + x
