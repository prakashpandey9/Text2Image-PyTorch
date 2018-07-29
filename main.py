import os
import torch
import train
import argparse
import numpy as np

from train import GAN_CLS
from torch.utils.data import DataLoader
from data_loader import Text2ImageDataset


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print ('{} created'.format(dir_name))


def check_args(args):
    # Make all directories if they don't exist

    # --checkpoint_dir
    check_dir(args.checkpoint_dir)

    # --sample_dir
    check_dir(args.sample_dir)

    # --log_dir
    check_dir(args.log_dir)

    # --final_model dir
    check_dir(args.final_model)

    # --epoch
    assert args.num_epochs > 0, 'Number of epochs must be greater than 0'

    # --batch_size
    assert args.batch_size > 0, 'Batch size must be greater than zero'

    # --z_dim
    assert args.z_dim > 0, 'Size of the noise vector must be greater than zero'

    return args


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument_group('Dataset related arguments')
    parser.add_argument('--data_dir', type=str, default="Data",
                        help='Data Directory')

    parser.add_argument('--dataset', type=str, default="flowers",
                        help='Dataset to train')

    parser.add_argument_group('Model saving path and steps related arguments')
    parser.add_argument('--log_step', type=int, default=100,
                        help='Save INFO into logger after every x iterations')

    parser.add_argument('--sample_step', type=int, default=100,
                        help='Save generated image after every x iterations')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Save model checkpoints after every x iterations')

    parser.add_argument('--sample_dir', type=str, default='sample',
                        help='Save generated image after every x iterations')

    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Save INFO into logger after every x iterations')

    parser.add_argument('--final_model', type=str, default='final_model',
                        help='Save INFO into logger after every x iterations')

    parser.add_argument_group('Model training related arguments')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Total number of epochs to train')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch Size')

    parser.add_argument('--img_size', type=int, default=64,
                        help='Size of the image')

    parser.add_argument('--z_dim', type=int, default=100,
                        help='Size of the latent variable')

    parser.add_argument('--text_embed_dim', type=int, default=4800,
                        help='Size of the embeddding for the captions')

    parser.add_argument('--text_reduced_dim', type=int, default=1024,
                        help='Reduced dimension of the caption encoding')

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Hyperparameter of the Adam optimizer')

    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Hyperparameter of the Adam optimizer')

    parser.add_argument('--l1_coeff', type=float, default=50,
                        help='Coefficient for the L1 Loss')

    parser.add_argument('--resume_epoch', type=int, default=1,
                        help='Resume epoch to resume training')

    args = parser.parse_args()

    check_args(args)

    dataset = Text2ImageDataset()
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    gan = GAN_CLS(args, data_loader)

    gan.build_model()
    gan.train_model()


if __name__ == '__main__':
    main()
