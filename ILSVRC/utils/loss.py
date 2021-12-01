import torch.nn.functional as F
import torch.nn as nn
import torch
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, output, labels):
        labels = labels.data.cpu()
        zero_ = torch.zeros(labels.size(0), 200)
        index = torch.unsqueeze(labels, 1)
        zero_.scatter_(1, index, 1)
        labels = zero_.float().cuda()

        loss_cls = F.multilabel_soft_margin_loss(output, labels)
        return loss_cls
