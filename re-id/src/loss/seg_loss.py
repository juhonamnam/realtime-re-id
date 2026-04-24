import torch

import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, predicted, target_mask):
        pt = torch.where(target_mask, predicted, 1 - predicted)
        pt = torch.clamp(pt, min=self.epsilon, max=1-self.epsilon)
        loss = - self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        return loss


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_seg, seg):
        batch = out_seg.size(0)
        height = out_seg.size(2)
        width = out_seg.size(3)

        fl = FocalLoss()

        seg_resized = F.interpolate(seg, size=(height, width), mode="nearest")
        seg_mask = seg_resized > 0.5

        f_loss = fl(out_seg, seg_mask)
        f_loss = torch.where(seg_mask, f_loss, 0)
        f_loss = torch.sum(f_loss) / batch

        return f_loss
