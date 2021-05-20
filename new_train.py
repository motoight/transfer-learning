import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset_1 import parse_root, Def_Dataset
import argparse
from random import shuffle
import math
from vgg_face import Vgg_face_dag
from xception import xception
from torch.hub import load_state_dict_from_url
from tqdm import tqdm
from utils import AverageMeter, get_mean_and_std
from sklearn.metrics import classification_report
import os
import shutil
import json
import wandb

class MLP_Classifier(nn.Module):
    '''
        一个两层MLP分类器
    '''
    def __init__(self, in_feature, hidden_feature, n_classes):
        super(MLP_Classifier, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.n_classes = n_classes
        self.linear1 = nn.Linear(in_features=self.in_feature, out_features=self.hidden_feature)
        self.linear2 = nn.Linear(in_features=self.hidden_feature, out_features=self.n_classes)
        self.bn1 = nn.BatchNorm1d(self.hidden_feature)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, input):
        out = self.linear1(input)
        out = self.bn1(out)
        out = F.relu(out)
        return self.linear2(out)

# 设置训练参数
wandb.init(project='Baby emotional detector')
parser = argparse.ArgumentParser(description='Baby emotional detector')
parser.add_argument('--arch', type=str, choices=['VGG', 'Inception', 'ResNet50','Xception'])
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--resume', type=str, help='path to the resume model')
parser.add_argument('--data-root', type=str)
parser.add_argument('--out-dir', type=str)
parser.add_argument('--fix-encoder', action='store_true')



args = parser.parse_args()

# 根据arch参数初始化网络结构
# 分别有VGG-16， InceptionNet, ResNet50三种结构
# 网络的编码器部分使用预训练模型，分类器使用Xavier初始化
if args.arch == 'VGG':
    model = Vgg_face_dag()
    state_dict = load_state_dict_from_url('http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth')
    model.load_state_dict(state_dict)
    # model.fc8 = nn.Sequential(
    #     nn.Linear(in_features=4096, out_features=512, bias=True),
    #     nn.BatchNorm1d(num_features=512),
    #     nn.Linear(in_features=512, out_features=args.n_classes)
    # )
    model.fc8 = MLP_Classifier(in_feature=4096, hidden_feature=512, n_classes=args.n_classes)

elif args.arch == 'Inception':
    model = models.inception_v3(pretrained=True, aux_logits=False)
    # model.fc = nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
    # model.fc = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=512, bias=True),
    #     nn.BatchNorm1d(num_features=512),
    #     nn.Linear(in_features=512, out_features=args.n_classes)
    # )
    model.fc = MLP_Classifier(in_feature=2048, hidden_feature=512, n_classes=args.n_classes)
elif args.arch == 'ResNet50':
    model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
    # model.fc = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=512, bias=True),
    #     nn.BatchNorm1d(num_features=512),
    #     nn.ReLU(),
    #     nn.Linear(in_features=512, out_features=args.n_classes)
    # )
    model.fc = MLP_Classifier(in_feature=2048, hidden_feature=512, n_classes=args.n_classes)
elif args.arch == 'Xception':
    model = xception(pretrained='imagenet')
    # model.last_linear = nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
    # model.last_linear = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=512, bias=True),
    #     nn.BatchNorm1d(num_features=512),
    #     nn.ReLU(),
    #     nn.Linear(in_features=512, out_features=args.n_classes)
    # )
    model.last_linear = MLP_Classifier(in_feature=2048, hidden_feature=512, n_classes=args.n_classes)
# 设置输出目录
args.out_dir='./out/'+args.arch
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
state_log = open(os.path.join(args.out_dir, 'test_acc.log'), 'w')

wandb.watch(model)
model.cuda()



# 初始化数据集
root = 'newborn2/newborn2/train' # 改成自己的目录
train_img_list, train_labels = parse_root(root)

root = 'newborn2/newborn2/test' # 改成自己的目录
test_img_list, test_labels = parse_root(root)

# 进行数据集划分
train_img_list, train_labels = np.array(train_img_list), np.array(train_labels)
test_img_list, test_labels = np.array(test_img_list), np.array(test_labels)
print("train len:{}".format(len(train_img_list)))
print("test len:{}".format(len(test_img_list)))

# # 抽样计算数据集的统计量
# dataset_statistics_path = 'dataset_statistics.json'
# if os.path.exists(dataset_statistics_path):
#     with open(dataset_statistics_path, 'r') as json_in:
#         dataset_statistics = json.load(json_in)
# else:
#     dataset_statistics = {}
#
# if not dataset_statistics.get(args.dataset):
#     idx = np.arange(len(train_img_list))
#     shuffle(idx)
#     # 采样10%的样本用于计算均值和方差统计量
#     sample_num = math.floor(0.1*len(train_img_list))
#     sampled_img_list, sampled_label = train_img_list[idx[:sample_num]], train_labels[idx[:sample_num]]
#     dataset = Def_Dataset(img_list=sampled_img_list, labels=sampled_label)
#     mean, std = get_mean_and_std(dataset)
#     print("Statistical mean:{mean}, std:{std}".format(mean=mean, std=std))
#     new_statistics = {
#         'mean': mean.numpy(),
#         'std': std.numpy()
#     }
#     dataset_statistics[args.dataset] = new_statistics
#     with open(dataset_statistics_path, 'w') as json_out:
#         json.dump(dataset_statistics, json_out)
# else:
#     mean = dataset_statistics[args.dataset]['mean']
#     std = dataset_statistics[args.dataset]['std']

# newborn 数据集统计量
mean = [0.3960, 0.2762, 0.2616]
std = [0.1026, 0.0965, 0.0950]
# imagenet 数据集统计量
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.1)),  # 功能：随机长宽比裁剪原始图片, 表示随机crop出来的图片会在的0.08倍至1.1倍之间
    transforms.RandomRotation(degrees=10),  # 功能：根据degrees随机旋转一定角度, 则表示在（-10，+10）度之间随机旋转
    transforms.ColorJitter(0.4, 0.4, 0.4),  # 功能：修改亮度、对比度和饱和度
    transforms.RandomHorizontalFlip(),  # 功能：水平翻转
    transforms.CenterCrop(size=224),  # 功能：根据给定的size从中心裁剪，size - 若为sequence,则为(h,w)，若为int，则(size,size)
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.1)),  # 功能：随机长宽比裁剪原始图片, 表示随机crop出来的图片会在的0.08倍至1.1倍之间
    transforms.CenterCrop(size=224),  # 功能：根据给定的size从中心裁剪，size - 若为sequence,则为(h,w)，若为int，则(size,size)
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


test_dataset = Def_Dataset(img_list=test_img_list, labels=test_labels,transform=transform_test)
train_dataset = Def_Dataset(img_list=train_img_list, labels=train_labels, transform=transform_train)
# 划分训练集为 trainset 和 evalset 按照8:2的比例
train_len = int(0.8 * len(train_dataset))
eval_len = len(train_dataset) - train_len
train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_len, eval_len])
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False, shuffle=True)
eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False)

train_losses = AverageMeter()
eval_losses = AverageMeter()
label_name = ['peace', 'cry', 'pain1', 'pain2'] # we give label 0 for 'nonpain' and label 1 for 'pain'

# 设置可训练网络层
trainable_layer = {
    'VGG': 'fc8',
    'Inception': 'fc',
    'ResNet50': 'fc',
    'Xception': 'last_linear'
}



# 对于数据量多且与源数据集分布差异较大的数据集，我们可以通过finetune 达到更好学习效果
if args.fix_encoder:
    optimizer=torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9)
    # 将非可训练层参数固定
    for name, p in model.named_parameters():
        if trainable_layer[args.arch] not in name:
            p.requires_grad = False
        print("{name}:{grad}".format(name=name, grad=p.requires_grad))
else:
    optimizer=torch.optim.SGD([
        {'params':[p for n, p in model.named_parameters() if trainable_layer[args.arch] not in n]},
        {'params':[p for n, p in model.named_parameters() if trainable_layer[args.arch] in n], 'lr': args.lr}
    ], lr=args.lr*0.05, momentum=0.9)


# 网络训练函数
def train(epoch, model, loader, optimizer):
    model.train()
    num_iter = (len(loader.dataset) // loader.batch_size) + 1
    # p_bar = tqdm(range(num_iter))
    for batch_idx,(imgs, labels) in enumerate(loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        output = model(imgs)
        loss = F.cross_entropy(output, labels, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item())
        wandb.log({'train_loss': train_losses.avg}, step=epoch+1)
        print("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}.  Loss: {loss:.4f}.".format(
                epoch=epoch + 1,
                epochs=args.num_epochs,
                batch=batch_idx + 1,
                iter=num_iter,
                loss=train_losses.avg))


# 网络验证函数
def eval(epoch, model, loader):
    model.eval()
    num_iter = (len(loader.dataset) // loader.batch_size) + 1
    correct = 0
    total = 0
    global best_acc
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            output = model(imgs)
            loss = F.cross_entropy(output, labels, reduction='mean')
            _, pred = torch.max(output,1)
            correct+=(pred==labels).sum()
            total+=labels.size(0)
            eval_losses.update(loss.item())
            print("Eval Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}.  Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.num_epochs,
                    batch=batch_idx + 1,
                    iter=num_iter,
                    loss=eval_losses.avg))
    acc = float(correct/total)
    wandb.log({'eval_loss': eval_losses.avg, 'eval_acc': acc}, step=epoch+1)
    checkpoint = {
        'epoch':epoch+1,
        'state_dict':model.state_dict(),
        'acc':best_acc,
        'optim_state':optimizer.state_dict()
    }
    # 按照验证集的最优准确率保存模型
    save_checkpoint(checkpoint, is_best=acc>best_acc, out_dir=args.out_dir)
    best_acc = max(best_acc, acc)
    print('Eval acc:{:.5f}'.format(acc))

# 网络测试函数
def test(epoch, model, loader):
    model.eval()
    num_iter = (len(loader.dataset) // loader.batch_size) + 1
    total_pred = []
    total_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            output = model(imgs)
            _, pred = torch.max(output,1)
            correct += (pred == labels).sum()
            total += labels.size(0)
            total_labels.extend(labels.cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
    print('Test Epoch{epoch}/{epochs:4}. Test acc:{acc:.5f}'.format(epoch=epoch + 1,
                    epochs=args.num_epochs, acc=float(correct / total)))
    state_log.write('Test Epoch{epoch}/{epochs:4}. Test acc:{acc:.5f}'.format(epoch=epoch + 1,
                    epochs=args.num_epochs, acc=float(correct / total)))

    wandb.log({'test_acc': float(correct / total)}, step=epoch+1)
    print(classification_report(y_true=total_labels, y_pred=total_pred, target_names=label_name))
    state_log.write(classification_report(y_true=total_labels, y_pred=total_pred, target_names=label_name))
    state_log.flush()

# 保存网络训练结果
def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(out_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(out_dir,
                                               'model_best.pth.tar'))
# 设置最优准确率
best_acc = 0
if args.resume:
    print("==> Resuming from checkpoint..")
    assert os.path.isfile(
        args.resume), "Error: no checkpoint directory found!"
    ckpt = torch.load(args.resume)
    args.start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optim_state'])
    best_acc = ckpt['acc']
    print('Last Eval acc is {}'.format(ckpt['acc']))


for epoch in range(args.start_epoch, args.num_epochs):
    train(epoch, model, train_loader, optimizer)
    if (epoch+1)%2==0:
        eval(epoch,model,eval_loader)
        test(epoch,model,test_loader)

state_log.close()