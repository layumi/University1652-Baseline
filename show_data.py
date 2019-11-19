import sys
import torch
import os
import numpy as np
from PIL import Image

target_root = 'data/train/drone'

def pad(inp, pad = 3):
    #print(inp.size)
    h, w = inp.size
    bg = np.zeros((h+2*pad, w+2*pad, len(inp.mode)))
    bg[pad:pad+h, pad:pad+w, :] = inp
    return bg

count = 0
ncol = 4
nrow = 4
npad = 3
im = {}
white_col = np.ones( (128+2*npad,24,3))*255
for folder_name in os.listdir(target_root):
    folder_root = target_root + '/' + folder_name
    if not os.path.isdir(folder_root):
        continue
    for img_name in os.listdir(folder_root):
        input1 = Image.open(folder_root + '/' + img_name)
        print(folder_root + '/' + img_name)
        input1 = input1.resize( (128, 128))
        # Start testing
        tmp = pad(input1, pad=npad)
        if count%ncol == 0:
            im[count//ncol] = tmp
        im[count//ncol] = np.concatenate((im[count//ncol], white_col, tmp), axis=1)
        count +=1
    if count > nrow*ncol:
        break
        

first_row = np.ones((128+2*npad,128+2*npad,3))*255
white_row = np.ones( (24,im[0].shape[1],3))*255
for i in range(nrow):
    if i == 0:
        pic = im[0]
    else:
        pic = np.concatenate((pic, im[i]), axis=0)
    pic = np.concatenate((pic, white_row), axis=0)
    #first_row = np.concatenate((first_row, white_col, im[i][0:256+2*npad, 0:256+2*npad, 0:3]), axis=1)

#pic = np.concatenate((first_row, white_row, pic), axis=0)
pic = Image.fromarray(pic.astype('uint8'))
#pic.save('sample_%s.jpg'%os.path.basename(target_root))
pic.save('sample.jpg')
