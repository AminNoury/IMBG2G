import torch
import torch.nn as nn
import torch.nn.functional as F


class ImbalanceLoss(nn.Module):
    def __init__(self, weight=None, use_focal=False, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.use_focal = use_focal
        self.gamma = gamma

    def forward(self, logits, labels):
        if not self.use_focal:
            return F.cross_entropy(logits, labels, weight=self.weight)

        ce_loss = F.cross_entropy(
            logits, labels, weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()
