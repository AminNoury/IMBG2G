import torch
import torch.nn as nn

class ImbalanceLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
      
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):

        loss = self.ce(logits, labels)
        return loss
