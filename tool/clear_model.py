import os

root = '../model/'
nn = []
for f in os.listdir(root):
    if not os.path.isdir(root+f):
        continue
    for ff in os.listdir(root+f):
        if ff[0:3] == 'net':
            if ff[5] =='a':
                continue
            if int(ff[4])<1:
                path = root+f+'/'+ff
                print(path)
                os.remove(path)
