import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataloader.Semi_dataloader import parse_root, Def_Dataset, parse_unlabel_root, Def_unlabel_Dataset
import argparse
from random import shuffle
import math
from models.vgg_face import Vgg_face_dag
from models.xception import xception
from torch.hub import load_state_dict_from_url
from tqdm import tqdm
from utils import AverageMeter
from sklearn.metrics import classification_report, confusion_matrix
import os
import shutil
import json
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torch import Tensor
import matplotlib

matplotlib.use('Agg')

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class MLP_Classifier(nn.Module):
    '''
        一个两层MLP分类器
    '''

    def __init__(self, in_feature, hidden_feature, n_classes, dropout=False):
        super(MLP_Classifier, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.n_classes = n_classes
        self.linear1 = nn.Linear(in_features=self.in_feature, out_features=self.hidden_feature)
        self.linear2 = nn.Linear(in_features=self.hidden_feature, out_features=self.n_classes)
        self.bn1 = nn.BatchNorm1d(self.hidden_feature)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, input):
        out = self.linear1(input)
        out = self.bn1(out)
        out = F.relu(out)
        if self.dropout:
            out = F.dropout(out, p=0.5)
        return self.linear2(out)


class SPLLoss(nn.NLLLoss):
    def __init__(self, *args, n_samples=0, threshold=0.1, growing_factor=1.3, **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)
        self.threshold = threshold
        self.growing_factor = growing_factor
        # v权重向量，v_i取值为{0,1}
        # v_i=0时表示样本不被选择（即为难样本），v_i=1时表示样本被选择（即为简单样本）
        self.v = torch.zeros(n_samples).int()

    def forward(self, input: Tensor, target: Tensor, index: Tensor) -> Tensor:
        super_loss = nn.functional.nll_loss(input, target, reduction="none")
        v = self.spl_loss(super_loss)
        self.v[index] = v.cpu()
        return (super_loss * v).mean(), v

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target, reduction='mean'):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if reduction == 'mean':
            return loss.mean()
        else:
            return loss

# 设置训练参数
# wandb.init(project='Baby emotional detector')
parser = argparse.ArgumentParser(description='Baby emotional detector')
parser.add_argument('--arch', type=str, choices=['VGG', 'Inception', 'ResNet50', 'Xception'])
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--resume', type=str, help='path to the resume model')
parser.add_argument('--data-root', type=str)
parser.add_argument('--out-dir', type=str)
parser.add_argument('--fix-encoder', action='store_true')
parser.add_argument('--use_dropout', action='store_true')
parser.add_argument('--use_labelsmooth', action='store_true')
parser.add_argument('--threshold', type=float, default=0.9) # threshold for design whether model prediction is reliable
parser.add_argument('--mu', type=int, default=7) # Hyper params to control the ratio take labeled and unlabeled data
parser.add_argument('--lambda_u', type=int, default=5) # Hyper params to control loss_x and loss_u
parser.add_argument('--print_interval', type=int, default=20) # print_interval=20 means in each epoch, print training msg each 20 step
args = parser.parse_args()
writer_comment = "_epochs={num_epochs}_fixed={fix_encoder}_smooth={use_labelsmooth}_bs={bs}_mu={mu}".format(
    num_epochs = args.num_epochs,
    fix_encoder = args.fix_encoder,
    use_labelsmooth = args.use_labelsmooth,
    bs = args.batch_size,
    mu = args.mu
)
writer = SummaryWriter(comment=writer_comment)

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
    model.fc8 = MLP_Classifier(in_feature=4096, hidden_feature=512, n_classes=args.n_classes, dropout=args.use_dropout)

elif args.arch == 'Inception':
    model = models.inception_v3(pretrained=True, aux_logits=False)
    # model.fc = nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
    # model.fc = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=512, bias=True),
    #     nn.BatchNorm1d(num_features=512),
    #     nn.Linear(in_features=512, out_features=args.n_classes)
    # )
    model.fc = MLP_Classifier(in_feature=2048, hidden_feature=512, n_classes=args.n_classes, dropout=args.use_dropout)
elif args.arch == 'ResNet50':
    model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
    # model.fc = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=512, bias=True),
    #     nn.BatchNorm1d(num_features=512),
    #     nn.ReLU(),
    #     nn.Linear(in_features=512, out_features=args.n_classes)
    # )
    model.fc = MLP_Classifier(in_feature=2048, hidden_feature=512, n_classes=args.n_classes, dropout=args.use_dropout)
elif args.arch == 'Xception':
    model = xception(pretrained='imagenet')
    # model.last_linear = nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
    # model.last_linear = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=512, bias=True),
    #     nn.BatchNorm1d(num_features=512),
    #     nn.ReLU(),
    #     nn.Linear(in_features=512, out_features=args.n_classes)
    # )
    model.last_linear = MLP_Classifier(in_feature=2048, hidden_feature=512, n_classes=args.n_classes,
                                       dropout=args.use_dropout)
# 设置输出目录
args.out_dir = './out/' + args.arch
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
state_log_name = 'test_acc' + writer_comment + '.log'
print("Test Results log in : {}".format(state_log_name))
state_log = open(os.path.join(args.out_dir, state_log_name), 'w')

model.cuda()

# 初始化有标签数据集
root = 'datasets/newborn2/newborn2/train'  # 改成自己的目录
train_img_list, train_labels = parse_root(root)

root = 'datasets/newborn2/newborn2/test'  # 改成自己的目录
test_img_list, test_labels = parse_root(root)

root = 'datasets/video_framemid/video_framemid/train' # 改成自己的目录
unlabel_img_list, unlabel_labels = parse_unlabel_root(root)


# 进行数据集划分
train_img_list, train_labels = np.array(train_img_list), np.array(train_labels)
test_img_list, test_labels = np.array(test_img_list), np.array(test_labels)
unlabel_img_list, unlabel_labels = np.array(unlabel_img_list), np.array(unlabel_labels)
print("train len:{}".format(len(train_img_list)))
print("test len:{}".format(len(test_img_list)))
print("unlabeled len:{}".format(len(unlabel_img_list)))


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
# mean = [0.3960, 0.2762, 0.2616]
# std = [0.1026, 0.0965, 0.0950]
# imagenet 数据集统计量
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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

# 引入强弱增广，通过弱增广作为强增广的伪标签进行无标签数据的学习
from randaugment import RandAugmentMC
class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224)
            ])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            RandAugmentMC(n=2, m=10)])

        self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

test_dataset = Def_Dataset(img_list=test_img_list, labels=test_labels, transform=transform_test)
train_dataset = Def_Dataset(img_list=train_img_list, labels=train_labels, transform=transform_train)
unlabeled_dataset = Def_unlabel_Dataset(img_list=unlabel_img_list, labels=unlabel_labels, transform=TransformFixMatch())

# splloss = SPLLoss(n_samples=len(train_dataset))
# 划分训练集为 trainset 和 evalset 按照8:2的比例
train_len = int(0.8 * len(train_dataset))
eval_len = len(train_dataset) - train_len
train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_len, eval_len])
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False,
                          shuffle=True)
eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False)
unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=args.batch_size * args.mu, num_workers=4, drop_last=False, shuffle=True)

print('Labeled batch size :{}. Unlabeled batch size :{}'.format(args.batch_size, args.batch_size*args.mu))

train_losses = AverageMeter()
train_lossesx = AverageMeter()
train_lossesu = AverageMeter()
Consis_loss = AverageMeter()
eval_losses = AverageMeter()
# 标签平滑项设置为0.1
labelsmoothLoss = LabelSmoothing(smoothing=0.1)
label_name = ['peace', 'cry', 'pain1', 'pain2']  # we give label 0 for 'nonpain' and label 1 for 'pain'

# 设置可训练网络层
trainable_layer = {
    'VGG': ['fc8', 'fc7', 'fc6'],
    'Inception': ['fc'],
    'ResNet50': ['fc'],
    'Xception': ['last_linear']
}

# 对于数据量多且与源数据集分布差异较大的数据集，我们可以通过finetune 达到更好学习效果
if args.fix_encoder:
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 将非可训练层参数固定
    for name, p in model.named_parameters():
        if not any(nd in name for nd in trainable_layer[args.arch]):
            p.requires_grad = False
        print("{name}:{grad}".format(name=name, grad=p.requires_grad))
else:
    # 迁移学习中网络参数分成正常训练的和微调两部分，分别设置不同的学习率
    # 其中 finetune 部分一般指网络前若干层卷积层，用于提取图像高层特征，用较小的学习率
    # 正常训练部分指网络靠近输出的若干卷积层和全连接层，设置正常学习率
    train_param = [p for n, p in model.named_parameters() if any(nd in n for nd in trainable_layer[args.arch])]
    finetune_param = [p for n, p in model.named_parameters() if not any(nd in n for nd in trainable_layer[args.arch])]

    optimizer = torch.optim.SGD([
        {'params': finetune_param},
        {'params': train_param, 'lr': args.lr}
    ], lr=args.lr * 0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 网络训练函数
def train(epoch, model, labeled_loader, unlabeled_loader, optimizer):
    model.train()
    num_iter = (len(labeled_loader.dataset) // labeled_loader.batch_size) + 1
    labeled_loader_iter = iter(labeled_loader)
    unlabeled_loader_iter = iter(unlabeled_loader)
    # p_bar = tqdm(range(num_iter))
    selected_sample_labels = []
    correct = 0
    total = 0
    for i in range(num_iter):
        try:
            input_x, labels, idx = labeled_loader_iter.next()
        except StopIteration:
            labeled_loader_iter = iter(labeled_loader)
            input_x, labels, idx = labeled_loader_iter.next()
        # 防止取unlabel data时没有数据
        try:
            input_uw, input_us, labels_u = unlabeled_loader_iter.next()
        except StopIteration:
            unlabeled_loader_iter = iter(unlabeled_loader)
            input_uw, input_us, labels_u = unlabeled_loader_iter.next()

        batch_size = input_x.size(0)
        inputs = torch.cat((input_x, input_uw, input_us))


        inputs = inputs.cuda()
        labels = labels.cuda()
        labels_u = labels_u.cuda()
        outputs = model(inputs)

        logits_x = outputs[:batch_size]
        logits_u_w, logits_u_s = outputs[batch_size:].chunk(2)

        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold)

        # if model prediction probability is higher than threshold, take the model prediction as pseudo label
        # else take the original label as label to simulate active learning phase
        refine_labels_u = torch.where(mask, targets_u, labels_u)

        # if args.use_labelsmooth:
        #     Lx = labelsmoothLoss(logits_x, labels, reduction='mean')
        #     Lu = torch.mean(labelsmoothLoss(logits_u_s, targets_u, reduction='none') * mask.float())
        # else:
        #     Lx = F.cross_entropy(logits_x, labels, reduction='mean')
        #     Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        if args.use_labelsmooth:
            Lx = labelsmoothLoss(logits_x, labels, reduction='mean')
            Lu = labelsmoothLoss(logits_u_w, refine_labels_u, reduction='mean')
        else:
            Lx = F.cross_entropy(logits_x, labels, reduction='mean')
            Lu = F.cross_entropy(logits_u_w, refine_labels_u , reduction='mean')
        # 一致性约束，对于unlabel样本通过强弱增广预测结果一致来约束网络稳定性
        consistency_loss = torch.mean((logits_u_s - logits_u_w)**2)


        _, pred = torch.max(logits_x, 1)
        correct += (pred == labels).sum()
        total += labels.size(0)

        loss = Lx + args.lambda_u * Lu + consistency_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item())
        train_lossesx.update(Lx.item())
        train_lossesu.update(Lu.item())
        Consis_loss.update(consistency_loss.item())
        if (i+1)%args.print_interval==0:
            print("Train Epoch: {epoch}/{epochs:4}. Step: {step}/{num_iter:4}. Loss: {loss:.4f}. Lossx: {lossx:.4f}. Lossu: {lossu:.4f}. Consis_loss: {consis_loss:.4f}.".format(
                epoch=epoch + 1,
                step=i+1,
                epochs=args.num_epochs,
                num_iter=num_iter,
                loss=train_losses.avg,
                lossx=train_lossesx.avg,
                lossu=train_lossesu.avg,
                consis_loss=Consis_loss.avg
            ))

    acc = float(correct / total)
    writer.add_scalar('Loss/train_loss', train_losses.avg, epoch + 1)
    writer.add_scalar('Loss/train_lossx', train_lossesx.avg, epoch + 1)
    writer.add_scalar('Loss/train_lossu', train_lossesu.avg, epoch + 1)
    writer.add_scalar('Loss/consistency_loss', Consis_loss.avg, epoch + 1)
    writer.add_scalar('Acc/train_acc', acc, epoch + 1)
    print("Train Epoch: {epoch}/{epochs:4}.  Loss: {loss:.4f}. Lossx: {lossx:.4f}. Lossu: {lossu:.4f}. Consis_loss: {consis_loss:.4f}."
          "Acc: {Acc:.4f}.".format(
        epoch=epoch + 1,
        epochs=args.num_epochs,
        loss=train_losses.avg,
        lossx=train_lossesx.avg,
        lossu=train_lossesu.avg,
        consis_loss = Consis_loss.avg,
        Acc=acc
    ))


    #
    # for batch_idx, (imgs, labels, idx) in enumerate(loader):
    #     imgs, labels = imgs.cuda(), labels.cuda()
    #     output = model(imgs)
    #     loss = F.cross_entropy(output, labels, reduction='mean')
    #     # loss, mask = splloss(output, labels, idx)
    #     # selected_sample_labels.extend(labels[mask.bool()].cpu().numpy())
    #     _, pred = torch.max(output, 1)
    #     correct += (pred == labels).sum()
    #     total += labels.size(0)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     train_losses.update(loss.item())
    #     # wandb.log({'train_loss': train_losses.avg}, step=epoch+1)
    #     print("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}.  Loss: {loss:.4f}.".format(
    #         epoch=epoch + 1,
    #         epochs=args.num_epochs,
    #         batch=batch_idx + 1,
    #         iter=num_iter,
    #         loss=train_losses.avg))
    # acc = float(correct / total)
    # writer.add_scalar('Loss/train_loss', train_losses.avg, epoch + 1)
    # writer.add_scalar('Acc/train_acc', acc, epoch + 1)
    #


    # from collections import Counter
    # print("selected_sample_labels:{}".format(Counter(selected_sample_labels)))
    # all_labels = np.array(selected_sample_labels)
    # writer.add_histogram('selected_sample_labels hist', all_labels)


# 网络验证函数
def eval(epoch, model, loader):
    model.eval()
    num_iter = (len(loader.dataset) // loader.batch_size) + 1
    correct = 0
    total = 0
    global best_acc
    with torch.no_grad():
        for batch_idx, (imgs, labels, _) in enumerate(loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            output = model(imgs)
            loss = F.cross_entropy(output, labels, reduction='mean')
            _, pred = torch.max(output, 1)
            correct += (pred == labels).sum()
            total += labels.size(0)
            eval_losses.update(loss.item())
            print("Eval Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}.  Loss: {loss:.4f}. ".format(
                epoch=epoch + 1,
                epochs=args.num_epochs,
                batch=batch_idx + 1,
                iter=num_iter,
                loss=eval_losses.avg))
    acc = float(correct / total)
    # wandb.log({'eval_loss': eval_losses.avg, 'eval_acc': acc}, step=epoch+1)
    writer.add_scalar('Loss/eval_loss', eval_losses.avg, epoch + 1)
    writer.add_scalar('Acc/eval_acc', acc, epoch + 1)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': best_acc,
        'optim_state': optimizer.state_dict()
    }
    # 按照验证集的最优准确率保存模型
    save_checkpoint(checkpoint, is_best=acc > best_acc, out_dir=args.out_dir)
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
        for batch_idx, (imgs, labels, _) in enumerate(loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            output = model(imgs)
            _, pred = torch.max(output, 1)
            correct += (pred == labels).sum()
            total += labels.size(0)
            total_labels.extend(labels.cpu().numpy())
            total_pred.extend(pred.cpu().numpy())
    print('Test Epoch{epoch}/{epochs:4}. Test acc:{acc:.5f}'.format(epoch=epoch + 1,
                                                                    epochs=args.num_epochs, acc=float(correct / total)))
    state_log.write('Test Epoch{epoch}/{epochs:4}. Test acc:{acc:.5f}'.format(epoch=epoch + 1,
                                                                              epochs=args.num_epochs,
                                                                              acc=float(correct / total)))
    acc = float(correct / total)
    # wandb.log({'test_acc': float(correct / total)}, step=epoch+1)
    writer.add_scalar('Acc/test_acc', acc, epoch + 1)
    print(classification_report(y_true=total_labels, y_pred=total_pred, target_names=label_name))
    #######################
    # add confusion_matrix
    confusion = confusion_matrix(total_pred, total_labels)
    fig = plt.figure(figsize=(10, 7))
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.xticks(range(4), ['peace', 'cry', 'pain1', 'pain2'], fontsize=8)
    plt.yticks(range(4), ['peace', 'cry', 'pain1', 'pain2'], fontsize=8)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    writer.add_figure('Fig/confusion_matrix', fig, epoch + 1)
    ######################
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
    train(epoch, model,train_loader, unlabeled_loader, optimizer)
    scheduler.step()
    print("lr = " + str(scheduler.get_last_lr()[0]))
    if (epoch + 1) % 2 == 0:
        eval(epoch, model, eval_loader)
        test(epoch, model, test_loader)

state_log.close()