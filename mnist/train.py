import os
import sys
import argparse
import gzip
import pickle
sys.path.append(os.getcwd())
from aae import AdversarialAutoEncoder as AAE
from image_sampler import ImageSampler
from noise_sampler import NoiseSampler
from utils.config import dump_config
from mnist.autoencoder import AutoEncoder
from mnist.discriminator import Discriminator


def load_mnist_train(file_path):
    f = gzip.open(file_path)
    (train_x, _), (_, _), (_, _) = pickle.load(f, encoding='latin1')
    train_x = train_x.reshape(len(train_x), 28, 28, 1)
    train_x = (train_x * 255).astype('uint8')
    return train_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mnist_path', type=str)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=1000)
    parser.add_argument('--latent_dim', '-ld', type=int, default=2)
    parser.add_argument('--height', '-ht', type=int, default=32)
    parser.add_argument('--width', '-wd', type=int, default=32)
    parser.add_argument('--channel', '-ch', type=int, default=1)
    parser.add_argument('--save_steps', '-ss', type=int, default=10)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=10)
    parser.add_argument('--model_dir', '-md', type=str, default="./params")
    parser.add_argument('--result_dir', '-rd', type=str, default="./result")
    parser.add_argument('--noise_mode', '-nm', type=str, default="normal")

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    dump_config(os.path.join(args.result_dir, 'config.csv'), args)
    input_shape = (args.height, args.width, args.channel)

    image_sampler = ImageSampler(target_size=(args.width, args.height),
                                 color_mode='rgb' if args.channel == 3 else 'gray',
                                 is_training=True)
    noise_sampler = NoiseSampler(args.noise_mode)

    autoencoder = AutoEncoder(input_shape, args.latent_dim,
                              is_training=True,
                              color_mode='rgb' if args.channel == 3 else 'gray')
    discriminator = Discriminator(is_training=True)

    aae = AAE(autoencoder, discriminator, is_training=True)

    aae.fit_generator(image_sampler.flow(load_mnist_train(args.mnist_path),
                                         batch_size=args.batch_size),
                      noise_sampler, nb_epoch=args.nb_epoch,
                      save_steps=args.save_steps, visualize_steps=args.visualize_steps,
                      result_dir=args.result_dir, model_dir=args.model_dir)


if __name__ == '__main__':
    main()
