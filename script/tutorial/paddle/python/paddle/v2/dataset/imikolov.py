# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
imikolov's simple dataset: http://www.fit.vutbr.cz/~imikolov/rnnlm/

Complete comments.
"""
import paddle.v2.dataset.common
import tarfile

__all__ = ['train', 'test', 'build_dict']

URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
MD5 = '30177ea32e27c525793142b6bf2c8e2d'


def word_count(f, word_freq=None):
    add = paddle.v2.dataset.common.dict_add
    if word_freq == None:
        word_freq = {}

    for l in f:
        for w in l.strip().split():
            add(word_freq, w)
        add(word_freq, '<s>')
        add(word_freq, '<e>')

    return word_freq


def build_dict():
    train_filename = './simple-examples/data/ptb.train.txt'
    test_filename = './simple-examples/data/ptb.valid.txt'
    with tarfile.open(
            paddle.v2.dataset.common.download(
                paddle.v2.dataset.imikolov.URL, 'imikolov',
                paddle.v2.dataset.imikolov.MD5)) as tf:
        trainf = tf.extractfile(train_filename)
        testf = tf.extractfile(test_filename)
        word_freq = word_count(testf, word_count(trainf))
        if '<unk>' in word_freq:
            # remove <unk> for now, since we will set it as last index
            del word_freq['<unk>']

        TYPO_FREQ = 50
        word_freq = filter(lambda x: x[1] > TYPO_FREQ, word_freq.items())

        word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*word_freq_sorted))
        word_idx = dict(zip(words, xrange(len(words))))
        word_idx['<unk>'] = len(words)

    return word_idx


def reader_creator(filename, word_idx, n):
    def reader():
        with tarfile.open(
                paddle.v2.dataset.common.download(
                    paddle.v2.dataset.imikolov.URL, 'imikolov',
                    paddle.v2.dataset.imikolov.MD5)) as tf:
            f = tf.extractfile(filename)

            UNK = word_idx['<unk>']
            for l in f:
                l = ['<s>'] + l.strip().split() + ['<e>']
                if len(l) >= n:
                    l = [word_idx.get(w, UNK) for w in l]
                    for i in range(n, len(l) + 1):
                        yield tuple(l[i - n:i])

    return reader


def train(word_idx, n):
    return reader_creator('./simple-examples/data/ptb.train.txt', word_idx, n)


def test(word_idx, n):
    return reader_creator('./simple-examples/data/ptb.valid.txt', word_idx, n)
