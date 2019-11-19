import os

drone_path = './data/train/drone'
sample = [1, 3, 9, 18, 27]
step_list   = [54, 18, 6, 3, 2]

for name, step in zip(sample, step_list):
    target_path = drone_path + str(name)
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    selected = range(1,54,step)
    for dir_name in os.listdir(drone_path):
        folder_name = drone_path + '/' + dir_name + '/'
        target_folder_name = target_path + '/' + dir_name + '/'
        if not os.path.isdir(folder_name):
            continue
        if not os.path.isdir(target_folder_name):
            os.mkdir(target_folder_name)
        for file_name in os.listdir(folder_name):
            number = file_name.split('-')
            number = number[1]
            number = int(number[0:2])
            if number%step == 1:
                os.system('cp %s %s'%(folder_name + file_name, target_folder_name + file_name ))
