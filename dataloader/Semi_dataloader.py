import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms
# parse data root and format dataset
# we give label 0 for 'nonpain' and label 1 for 'pain'
import os
from PIL import Image
from random import shuffle
import math

# 解析数据集文件
def parse_root(root):
    '''
    :param root: 数据集根目录
    :return:
        img_list: list 图片路径
        labels: list 对应标签
    '''
    root=root.replace('\\', '/')
    _list = os.listdir(root)
    img_list = []
    labels = []


    peace_num = 0
    cry_num = 0
    pain1_num = 0
    pain2_num = 0


    for item in _list:
        # 根据数据集文件中的txt文件，得到图片的路径和相应的标签
        # 我们把nonpain的图片标注为0， pain的图片标注为1
        if item.endswith('.txt'):
            fpath = os.path.join(root, item)
            with open(fpath, 'r') as ff:
                lines = ff.readlines()
                for line in lines:
                    line = line.split(' ')[0]
                    if 'peace' in line:
                        img_list.append(os.path.join(root, line.strip()))
                        labels.append(0)
                        peace_num+=1
                    if 'cry' in line:
                        img_list.append(os.path.join(root, line.strip()))
                        labels.append(1)
                        cry_num+=1
                    if 'pain1' in line:
                        img_list.append(os.path.join(root, line.strip()))
                        labels.append(2)
                        pain1_num+=1
                    if 'pain2' in line:
                        img_list.append(os.path.join(root, line.strip()))
                        labels.append(3)
                        pain2_num+=1
    assert peace_num + cry_num + pain1_num + pain2_num == len(img_list)
    print(f"peace: {peace_num}, cry: {cry_num}, pain1: {pain1_num}, pain2: {pain2_num}")
    # 得到了图片的路径list和对应的label list
    return img_list, labels

def parse_unlabel_root(root):
    root = root.replace('\\', '/')
    _list = os.listdir(root)
    img_list = []
    labels = []
    peace_num = 0
    cry_num = 0
    pain1_num = 0
    pain2_num = 0
    subdirs = []
    name2labels = []
    for item in _list:
        # 统计各个类别图片的数量和子目录路径
        subdir = os.path.join(root, item)
        subdirs.append(subdir)
        sublist = os.listdir(subdir)
        if item == '1peace':
            label = 0
            peace_num = len(sublist)
        elif item == '2cry':
            label = 1
            cry_num = len(sublist)
        elif item == '3pain1':
            label = 2
            pain1_num = len(sublist)
        elif item == '4pain2':
            label = 3
            pain2_num = len(sublist)
        name2labels.append(label)

    print(f"peace: {peace_num}, cry: {cry_num}, pain1: {pain1_num}, pain2: {pain2_num}")
    min_len = min([peace_num, cry_num, pain1_num, pain2_num])
    print(f"To balance image num， Align to the same num：{min_len}")
    # print(name2labels)
    # print(subdirs)
    for i in range(4):
        sublist = os.listdir(subdirs[i])
        from random import shuffle
        shuffle(sublist)
        sublist = sublist[:min_len]
        for subitem in sublist:
            if subitem.endswith('.jpg'):
                img_list.append(os.path.join(subdirs[i], subitem))
                labels.append(name2labels[i])
    assert 4*min_len == len(img_list)
    assert len(img_list) == len(labels)
    print(f"Total unlabel img num: {len(img_list)}")
    # 得到了图片的路径list和对应的label list
    return img_list, labels



# COPE数据集
class Def_Dataset(Dataset):
    def __init__(self, img_list, labels, transform=None):
        self.img_list = img_list
        self.labels = labels
        self.transform = transform if transform else transforms.Compose([
                                    transforms.ToTensor()])


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        label = self.labels[idx]

        img = self.transform(img)
        return img, label, idx

    def __len__(self):
        return len(self.img_list)


class Def_unlabel_Dataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        imgw, imgs = self.transform(img)
        return imgw, imgs

    def __len__(self):
        return len(self.img_list)



if __name__ == '__main__':
    root = './datasets/newborn2/newborn2/test'  # 改成自己的目录
    train_img_list, train_labels = parse_root(root)
    train_img_list, train_labels = np.array(train_img_list), np.array(train_labels)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.1)),  # 功能：随机长宽比裁剪原始图片, 表示随机crop出来的图片会在的0.08倍至1.1倍之间
        transforms.CenterCrop(size=224),  # 功能：根据给定的size从中心裁剪，size - 若为sequence,则为(h,w)，若为int，则(size,size)
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_set = Def_Dataset(img_list=train_img_list, labels=train_labels, transform=transform_test)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=train_set, batch_size=257, shuffle=True, num_workers=4)
    num_iter = (len(dataloader.dataset))//dataloader.batch_size + 1
    dataloader_iter = iter(dataloader)
    for i in range(num_iter+10):
        try:
            input_x, labels, idx = dataloader_iter.next()
        except:
            dataloader_iter = iter(dataloader)
            input_x, labels, idx = dataloader_iter.next()
        print(i)


    # idx = np.arange(len(train_img_list))
    # shuffle(idx)
    # # 采样10%的样本用于计算均值和方差统计量
    # sample_num = math.floor(0.1 * len(train_img_list))
    # sampled_img_list, sampled_label = train_img_list[idx[:sample_num]], train_labels[idx[:sample_num]]
    # dataset = Def_Dataset(img_list=train_img_list, labels=train_labels)
    # mean, std = get_mean_and_std(dataset)
    # print("Statistical mean:{mean}, std:{std}".format(mean=mean, std=std))




