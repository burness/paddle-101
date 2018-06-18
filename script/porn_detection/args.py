import argparse

__all__ = ['parse_args', ]

def parse_args():
    parser = argparse.ArgumentParser('Fluid Resnet')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    #  args related to learning rate
    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help='The learning rate.')
    parser.add_argument(
        '--iterations', type=int, default=80, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='If gpus > 1, will use ParallelExecutor to run, else use Executor.')
    # this option is available only for vgg and resnet.
    parser.add_argument(
        '--cpus',
        type=int,
        default=1,
        help='If cpus > 1, will use ParallelDo to run, else use Executor.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    parser.add_argument(
        '--no_test',
        action='store_true',
        help='If set, do not test the testset during training.')
    parser.add_argument(
        '--memory_optimize',
        action='store_true',
        help='If set, optimize runtime memory before start.')
    parser.add_argument(
        '--use_fake_data',
        action='store_true',
        help='If set ommit the actual read data operators.')
    parser.add_argument(
        '--profile', action='store_true', help='If set, profile a few steps.')
    parser.add_argument(
        '--update_method',
        type=str,
        default='local',
        choices=['local', 'pserver', 'nccl2'],
        help='Choose parameter update method, can be local, pserver, nccl2.')
    parser.add_argument(
        '--no_split_var',
        action='store_true',
        default=False,
        help='Whether split variables into blocks when update_method is pserver')
    parser.add_argument(
        '--async_mode',
        action='store_true',
        default=False,
        help='Whether start pserver in async mode to support ASGD')
    parser.add_argument(
        '--use_reader_op',
        action='store_true',
        help='Whether to use reader op, and must specify the data path if set this to true.'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default="",
        help='Directory that contains all the training recordio files.')
    args = parser.parse_args()
    return args