from tensorflow.contrib.keras.python.keras.datasets import mnist


def load_mnist(mode='training'):
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    if mode == 'training':
        return train_x.reshape(list(train_x.shape) + [1]), train_y
    else:
        return test_x.reshape(list(test_x.shape) + [1]), test_y
