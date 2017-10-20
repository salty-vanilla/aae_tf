import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('agg')
sys.path.append(os.getcwd())
import seaborn as sns
import matplotlib.pyplot as plt
from aae import AdversarialAutoEncoder as AAE
from image_sampler import ImageSampler
from mnist.autoencoder import AutoEncoder
from mnist.discriminator import Discriminator
from mnist.dataset import load_mnist


def main():
    parser = argparse.ArgumentParser()
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

    test_x, test_y = load_mnist(mode='test')
    image_sampler = ImageSampler(target_size=(args.width, args.height),
                                 color_mode='rgb' if args.channel == 3 else 'gray',
                                 is_training=False)
    encoded = aae.predict_latent_vectors_generator(image_sampler.flow(test_x, shuffle=False))

    df = pd.DataFrame({'z_1': encoded[:, 0],
                       'z_2': encoded[:, 1],
                       'label': test_y})
    df.plot(kind='scatter', x='z_1', y='z_2',
            c='label', cmap='Set1', s=10)
    plt.savefig(os.path.join(args.result_dir, 'scatter.png'))


if __name__ == '__main__':
    main()
