import os
import sys
import argparse
import pickle
import gzip
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('agg')
sys.path.append(os.getcwd())
import seaborn as sns
import matplotlib.pyplot as plt
from aae import AdversarialAutoEncoder as AAE
from mnist.autoencoder import AutoEncoder
from mnist.discriminator import Discriminator


def load_mnist_test(file_path, target_size=None):
    f = gzip.open(file_path)
    (train_x, _), (valid_x, _), (test_x, test_y) = pickle.load(f, encoding='latin1')

    test_images = [Image.fromarray(x.reshape(28, 28)) for x in test_x]

    if target_size is not None and target_size != (28, 28):
        test_images = [x.resize(target_size, Image.BILINEAR)
                       for x in test_images]
    test_x = np.array([np.asarray(x) for x in test_images])
    test_x = test_x.reshape(len(test_x), target_size[0], target_size[1], 1)
    return test_x, test_y


def plot_scatter():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mnist_path', type=str)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--latent_dim', '-ld', type=int, default=2)
    parser.add_argument('--height', '-ht', type=int, default=32)
    parser.add_argument('--width', '-wd', type=int, default=32)
    parser.add_argument('--channel', '-ch', type=int, default=1)
    parser.add_argument('--model_path', '-mp', type=str, default="./params")
    parser.add_argument('--result_dir', '-rd', type=str, default="./result")

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    input_shape = (args.height, args.width, args.channel)

    autoencoder = AutoEncoder(input_shape, args.latent_dim,
                              is_training=False,
                              color_mode='rgb' if args.channel == 3 else 'gray')
    discriminator = Discriminator(is_training=False)

    aae = AAE(autoencoder, discriminator, is_training=False)
    aae.restore(args.model_path)

    test_x, test_y = load_mnist_test(args.mnist_path,
                                     (input_shape[1], input_shape[0]))
    encoded = aae.predict_latent_vectors(test_x)

    df = pd.DataFrame({'z_1': encoded[:, 0],
                       'z_2': encoded[:, 1],
                       'label': test_y})
    df.plot(kind='scatter', x='z_1', y='z_2',
            c='label', cmap='Set1', s=10)
    plt.savefig(os.path.join(args.result_dir, 'scatter.png'))


if __name__ == '__main__':
    main()
