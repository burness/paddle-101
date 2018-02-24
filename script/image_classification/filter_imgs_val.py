from paddle.v2.image import load_and_transform
import paddle.v2 as paddle

def filter_imgs(file_path = "val.lst", write_file = "valid_val.lst"):
    fwrite = open(write_file, "w")
    with open(file_path, 'r') as fread:
        error=0
        for line in fread.readlines():
            img_path = line.strip().split("\t")[0]
            try:
                img = paddle.image.load_image(img_path)
                img = paddle.image.simple_transform(img, 256, 224, True)
                fwrite.write(line)
            except:
                error += 1
                print error



filter_imgs()
