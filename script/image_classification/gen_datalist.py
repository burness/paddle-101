import random
import os
import codecs
import sys

def gen_datalist(data_dir, class_label, data_type="train", shuffle=True, suffix_list=["jpg", "jpeg", "JPEG", "jpg"]):
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        print "processing {0}".format(root)
        for file_name in files:
            file_name = os.path.join(root, file_name)
            suffix = file_name.split(".")[-1]
            if suffix in suffix_list:
                all_files.append(file_name) 
    if shuffle:
        print "shuffle now"
        random.shuffle(all_files)
    print "begin to write to {0}".format(data_type+"_"+class_label+".lst")
    with codecs.open(data_type+"_"+class_label+".lst", "w", encoding="utf8") as fwrite: 
        for each_file in all_files:
            fwrite.write(each_file+"\t"+class_label+"\n") 



if __name__ == "__main__":
    argv = sys.argv
    data_dir = argv[1]
    class_label = argv[2] 
    gen_datalist(data_dir, class_label)
