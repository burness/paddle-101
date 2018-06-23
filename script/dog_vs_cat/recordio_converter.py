import os
import random
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.dataset import  image
from multiprocessing import cpu_count
import functools
from PIL import Image, ImageEnhance
import math
import numpy as np
DATA_DIM=224
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio
    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * random.uniform(scale_min,
                                                             scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = random.randint(0, img.size[0] - w)
    j = random.randint(0, img.size[1] - h)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def rotate_image(img):
    angle = random.randint(-10, 10)
    img = img.rotate(angle)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


def process_image(sample, mode, color_jitter=False, rotate=False):
    img_path = sample[0]

    img = Image.open(img_path)
    #img = sample[0]
    if mode == 'train':
        if rotate: img = rotate_image(img)
        img = random_crop(img, DATA_DIM)
    else:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=DATA_DIM, center=True)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]

# mapper = functools.partial(process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

def default_mapper(sample):
    img, label = sample
    img = image.simple_transform(
        img, 256, 224, True, mean=[103.94, 116.78, 123.68])
    return img.flatten().astype('float32'), label


def dataset(data_dir, train_val_ratio=0.8):
    img_list = []
    img2label = dict()
    label2id = dict()
    sub_dirs = [i for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,i))]
    print(sub_dirs)
    for index, sub_dir in enumerate(sub_dirs):
        label2id[sub_dir] = index
        sub_files = []
        for root, dir, files in os.walk(os.path.join(data_dir, sub_dir)):
            print("files num: {0}, files top 5: {1}".format(len(files), files[:5]))
            sub_files += [os.path.join(root, tmp_file) for tmp_file in files  if tmp_file.split(".")[-1] in ["jpg", "jpeg", "JPG","JPEG"]]
        print("subdir name: {0}, file num: {1}".format(sub_dir, len(sub_files)))
        img_list += sub_files
        for file in sub_files:
            img2label[file] =sub_dir
    random.shuffle(img_list)
    train_len = int(train_val_ratio*len(img_list))
    train_img_list = img_list[:train_len]
    val_img_list = img_list[train_len:]
    print(len(train_img_list))

    def train_reader():
        for idx, imgfile in enumerate(train_img_list):
            try:
                #data = image.load_image(imgfile)
                label = [label2id[img2label[imgfile]], ]
                #h,w = data.shape[:2]
                yield [imgfile, label]
            except Exception as e:
                continue

    def test_reader():
        for idx, imgfile in enumerate(val_img_list):
            try:
                #data = image.load_image(imgfile)
                label = [label2id[img2label[imgfile]], ]
                #h,w = data.shape[:2]
                yield [imgfile, label]
            except Exception as e:
                print "error infor: {0}".format(e.message)
                continue
    train_mapper = functools.partial(process_image, mode="train", color_jitter=False, rotate=False)
    test_mapper = functools.partial(process_image, mode="val")
    #return paddle.reader.map_readers(default_mapper, train_reader), paddle.reader.map_readers(default_mapper, test_reader)
    return paddle.reader.xmap_readers(train_mapper, train_reader, cpu_count(), 51200), paddle.reader.xmap_readers(test_mapper, test_reader, cpu_count(), 5120)


