import torch.nn as nn
import torch


# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2, weight=None):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        self.weight = weight

    def _toOneHot_smooth(self, label, batchsize, classes):
        base_prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + base_prob

        if self.weight is not None:
            one_hot_label = one_hot_label * self.weight

        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)

        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        pre_logsoftmax = self.logSoftmax(pre)

        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)

        loss = torch.sum(-one_hot_label * pre_logsoftmax, dim=1)

        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)
