import os
from shutil import copyfile

download_path = 'data/cvpr2017_cvusa/'
train_split = download_path + 'splits/train-19zl.csv'
train_save_path = download_path + 'train/'

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(train_save_path + 'street')
    os.mkdir(train_save_path + 'satellite')

with open(train_split) as fp:
    line = fp.readline()
    while line:
        filename = line.split(',')
        #print(filename[0])
        src_path = download_path + '/' + filename[0]
        dst_path = train_save_path + '/satellite/' + os.path.basename(filename[0][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + os.path.basename(filename[0]))

        src_path = download_path + '/' + filename[1]
        dst_path = train_save_path + '/street/' + os.path.basename(filename[1][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + os.path.basename(filename[1]))

        line = fp.readline()


val_split = download_path + 'splits/val-19zl.csv'
val_save_path = download_path + 'val/'

if not os.path.isdir(val_save_path):
    os.mkdir(val_save_path)
    os.mkdir(val_save_path + 'street')
    os.mkdir(val_save_path + 'satellite')

with open(val_split) as fp:
    line = fp.readline()
    while line:
        filename = line.split(',')
        #print(filename[0])
        src_path = download_path + '/' + filename[0]
        dst_path = val_save_path + '/satellite/' + os.path.basename(filename[0][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + os.path.basename(filename[0]))

        src_path = download_path + '/' + filename[1]
        dst_path = val_save_path + '/street/' + os.path.basename(filename[1][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + os.path.basename(filename[1]))

        line = fp.readline()


