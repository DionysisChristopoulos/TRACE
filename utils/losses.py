import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        p = torch.sigmoid(input)
        p = torch.where(target >= 0.5, p, 1-p)

        modulating_factor = (1 - p)**self.gamma

        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal_loss = alpha * modulating_factor * bce_loss

        if self.size_average == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()