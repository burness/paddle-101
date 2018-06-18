from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import time
import os

import cProfile, pstats, StringIO

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
from recordio_converter import imagenet_dataset
from visualdl import LogWriter


def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
    conv1 = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv1, act=act)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]  # if args.data_format == 'NCHW' else input.shape[-1]
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_out, stride):
    short = shortcut(input, ch_out, stride)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def bottleneck(input, ch_out, stride):
    short = shortcut(input, ch_out * 4, stride)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv3, act='relu')


def layer_warp(block_func, input, ch_out, count, stride):
    res_out = block_func(input, ch_out, stride)
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1)
    return res_out


def resnet_imagenet(input, class_dim, depth=50, data_format='NCHW'):

    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_bn_layer(input, ch_out=64, filter_size=7, stride=2, padding=3)
    pool1 = fluid.layers.pool2d(
        input=conv1, pool_type='avg', pool_size=3, pool_stride=2)
    res1 = layer_warp(block_func, pool1, 64, stages[0], 1)
    res2 = layer_warp(block_func, res1, 128, stages[1], 2)
    res3 = layer_warp(block_func, res2, 256, stages[2], 2)
    res4 = layer_warp(block_func, res3, 512, stages[3], 2)
    pool2 = fluid.layers.pool2d(
        input=res4,
        pool_size=7,
        pool_type='avg',
        pool_stride=1,
        global_pooling=True)
    out = fluid.layers.fc(input=pool2, size=class_dim, act='softmax')
    return out

def train(args):
    logger = LogWriter(args.logdir, sync_cycle=10000)
    model = resnet_imagenet
    class_dim = 1000
    if args.data_format == 'NCHW':
        dshape = [3, 224, 224]
    else:
        dshape = [224, 224, 3]
    model = resnet_imagenet
    if not args.data_path:
        raise Exception(
            "Must specify --data_path when training with imagenet")
    train_reader, test_reader = imagenet_dataset(args.data_path)
    input = fluid.layers.data(name='image', shape=dshape, dtype='float32')
    predict = model(input, class_dim)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    batch_acc = fluid.layers.accuracy(input=predict, label=label)

    train_program = [avg_cost, batch_acc]

    optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)

    batched_train_reader = paddle.batch(
        paddle.reader.shuffle(
            train_reader, buf_size=5120),
        batch_size=args.batch_size * args.gpus,
        drop_last=True)
    batched_test_reader = paddle.batch(
        train_reader, batch_size=args.batch_size, drop_last=True)
    
    def event_handler(event):
         if isinstance(event, fluid.EndStepEvent):
            avg_cost, accuracy = trainer.test(
                reader=batched_test_reader, feed_order=['image', 'label'])
            print('Loss {0:2.2}, Acc {1:2.2}'.format(avg_cost, accuracy))
            # write the loss, acc to visualdl file
        if isinstance(event, fluid.EndEpochEvent):
            # save model to dir
            trainer.save_params(args.params_dirname)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    trainer = fluid.Trainer(
        train_func=train_program, optimizer_func=optimizer, place=place)

    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=['image', 'label'])


    
