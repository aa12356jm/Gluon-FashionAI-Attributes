'''
说明：此模块的主要作用是将原来给定的训练数据，按照类别数进行重新组织
'''


import mxnet
from mxnet import gluon, image
import os, shutil, random

# Read label.csv
# For each task, make folders, and copy picture to corresponding folders

label_dir = 'data/base/Annotations/label.csv'   #训练数据的所有图像文件名和对应标签
warmup_label_dir = 'data/web/Annotations/skirt_length_labels.csv' 

#训练数据一共有八种类别，对每一种类别都有多个属性，所以每个类型都要训练一个分类器来分类属性
label_dict = {'coat_length_labels': [],
              'lapel_design_labels': [],
              'neckline_design_labels': [],
              'skirt_length_labels': [],
              'collar_design_labels': [],
              'neck_design_labels': [],
              'pant_length_labels': [],
              'sleeve_length_labels': []}

task_list = label_dict.keys()

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

'''
打开文件，按行读取
'''
with open(label_dir, 'r') as f:
    lines = f.readlines()   #读取所有行
    tokens = [l.rstrip().split(',') for l in lines]  #对每一行进行分割，将分割结果保存在tokens中
    for path, task, label in tokens:  #每一行包含：图像路径，任务类别，标签
        label_dict[task].append((path, label))

mkdir_if_not_exist(['data/train_valid'])
mkdir_if_not_exist(['submission'])

for task, path_label in label_dict.items(): #遍历每个任务类别
    mkdir_if_not_exist(['data/train_valid',  task])  #对每个任务建立一个文件夹
    train_count = 0
    n = len(path_label) #path_label是保存的每个task中的图像和标签总数
    m = len(list(path_label[0][1])) #当前task中的属性数量，也就是分类几个属性

    #每个task中的属性数量，也就是类别数量，建立从0-N的文件夹
    for mm in range(m):
        mkdir_if_not_exist(['data/train_valid', task, 'train', str(mm)])
        mkdir_if_not_exist(['data/train_valid', task, 'val', str(mm)])

    random.shuffle(path_label)#随机打乱顺序
    for path, label in path_label: #对于每个图像和标签
        label_index = list(label).index('y') #每个图像对应的标签中的y的位置
        src_path = os.path.join('data/base', path) #每个图像的路径
        #将90%的数据用作训练
        if train_count < n * 0.9:
            shutil.copy(src_path,
                        os.path.join('data/train_valid', task, 'train', str(label_index)))
        else:#剩余的10%用作验证集
            shutil.copy(src_path,
                        os.path.join('data/train_valid', task, 'val', str(label_index)))
        train_count += 1

# Add warmup data to skirt task
#和上面的一样，针对skirt_length_labels类别，有一些额外的数据，将这些数据也拷贝过来作为训练数据
label_dict = {'skirt_length_labels': []}

with open(warmup_label_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        label_dict[task].append((path, label))

for task, path_label in label_dict.items():
    train_count = 0
    n = len(path_label)
    m = len(list(path_label[0][1]))

    random.shuffle(path_label)
    for path, label in path_label:
        label_index = list(label).index('y')
        src_path = os.path.join('data/web', path)
        if train_count < n * 0.9:
            shutil.copy(src_path,
                        os.path.join('data/train_valid', task, 'train', str(label_index)))
        else:
            shutil.copy(src_path,
                        os.path.join('data/train_valid', task, 'val', str(label_index)))
        train_count += 1

