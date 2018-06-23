import os
import random
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core





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
                print label, data
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

