# 工具类函数文件
import torch
from torch import Tensor
import torch.nn as nn

class AverageMeter(object):

    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))

    return mean, std



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
