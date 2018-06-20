import argparse

__all__ = ['parse_args', ]

def parse_args():
    parser = argparse.ArgumentParser('Fluid Resnet')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--class_dim', type=int, default=3, help='class num.')
    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help='The learning rate.')
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--data_path',
        type=str,
        default="",
        help='Directory that contains all the training recordio files.')
    args = parser.parse_args()
    return args
