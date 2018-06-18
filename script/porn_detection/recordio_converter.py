import os
import random
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.dataset import mnist, cifar, flowers, image


def convert_2_recordio(py_reader, outfilepath, batch_size, shape_data,
                       shape_label):
    num_batches = 0
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        reader = paddle.batch(py_reader(), batch_size=batch_size)
        feeder = fluid.DataFeeder(
            feed_list=[  # order is image and label
                fluid.layers.data(
                    name='image', shape=shape_data),
                fluid.layers.data(
                    name='label', shape=shape_label, dtype='int64'),
            ],
            place=fluid.CPUPlace())
        num_batches = fluid.recordio_writer.convert_reader_to_recordio_file(
            outfilepath, reader, feeder)
    return num_batches


def prepare_mnist(outpath, batch_size):
    outfilepath = os.path.join(outpath, "mnist.recordio")
    convert_2_recordio(mnist.train, outfilepath, batch_size, [784], [1])


def prepare_cifar10(outpath, batch_size):
    outfilepath = os.path.join(outpath, "cifar.recordio")
    convert_2_recordio(cifar.train10, outfilepath, batch_size, [3, 32, 32], [1])


def prepare_flowers(outpath, batch_size):
    outfilepath = os.path.join(outpath, "flowers.recordio")
    convert_2_recordio(flowers.train, outfilepath, batch_size, [3, 224, 224],
                       [1])


def default_mapper(sample):
    img, label = sample
    img = image.simple_transform(
        img, 256, 224, True, mean=[103.94, 116.78, 123.68])
    return img.flatten().astype('float32'), label


def imagenet_dataset(data_dir, train_val_ratio=0.8):
    img_list = []
    img2label = dict()
    label2id = dict()
    sub_dirs = [i for i in os.listdir(data_dir) if os.path.isdir(i)]
    for index, sub_dir in enumerate(sub_dirs):
        label2id[sub_dir] = index
        sub_abs_path = os.path.join(data_dir, sub_dir)
        sub_files = [os.path.join(sub_abs_path, i) for i in os.listdir(sub_abs_path) if i.split(".")[-1] in ["jpg, jpeg"]]
        img_list += sub_files
        for file in sub_files:
            img2label[file] =sub_dir
    random.shuffle(img_list)
    train_len = int(train_val_ratio*len(img_list))
    train_img_list = img_list[:train_len]
    val_img_list = img_list[train_len:]

    def train_reader():
        for idx, imgfile in enumerate(train_img_list):
            try:
                data = image.load_image(imgfile)
                label = [label2id[img2label[imgfile]], ]
                yield [data, label]
            except Exception as e:
                print "error infor: {0}".format(e.message)
                continue

    def test_reader():
        for idx, imgfile in enumerate(val_img_list):
            try:
                data = image.load_image(imgfile)
                label = [label2id[img2label[imgfile]], ]
                yield [data, label]
            except Exception as e:
                print "error infor: {0}".format(e.message)
                continue

    return paddle.reader.map_readers(default_mapper, train_reader), paddle.reader.map_readers(default_mapper, test_reader)



# FIXME(wuyi): delete this when https://github.com/PaddlePaddle/Paddle/pull/11066 is merged
def convert_reader_to_recordio_files(
        filename,
        batch_per_file,
        reader_creator,
        feeder,
        compressor=core.RecordIOWriter.Compressor.Snappy,
        max_num_records=1000,
        feed_order=None):
    if feed_order is None:
        feed_order = feeder.feed_names
    f_name, f_ext = os.path.splitext(filename)
    assert (f_ext == ".recordio")

    lines = []
    f_idx = 0
    counter = 0
    for idx, batch in enumerate(reader_creator()):
        lines.append(batch)
        if idx >= batch_per_file and idx % batch_per_file == 0:
            filename = "%s-%05d%s" % (f_name, f_idx, f_ext)
            with fluid.recordio_writer.create_recordio_writer(
                    filename, compressor, max_num_records) as writer:
                for l in lines:
                    res = feeder.feed(l)
                    for each in feed_order:
                        writer.append_tensor(res[each])
                    writer.complete_append_tensor()
                    counter += 1
                lines = []
                f_idx += 1
            print("written file: ", filename)
    return counter


def prepare_imagenet(inpath, outpath, batch_size):
    r = paddle.batch(imagenet_train(inpath), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        feed_list=[
            fluid.layers.data(
                name="image", shape=[3, 224, 224]), fluid.layers.data(
                    name="label", shape=[1], dtype='int64')
        ],
        place=fluid.CPUPlace())
    outpath = os.path.join(outpath, "imagenet.recordio")
    convert_reader_to_recordio_files(outpath, 10000, r, feeder)

