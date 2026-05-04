import torch

import torch.nn.functional as F
import torch.nn as nn

__all__ = ['SegLoss']

class FocalLoss(nn.Module):
    """Implementation of Focal Loss for binary classification.

    Focal loss is designed to address class imbalance by down-weighting well-classified
    examples and focusing on hard examples.

    Attributes:
        alpha (float): Weighting factor for the classes.
        gamma (float): Focusing parameter for hard examples.
        epsilon (float): Small value for numerical stability.
    """
    def __init__(self, alpha=1, gamma=2, epsilon=1e-8):
        """Initializes FocalLoss.

        Args:
            alpha (float, optional): Alpha parameter. Defaults to 1.
            gamma (float, optional): Gamma parameter. Defaults to 2.
            epsilon (float, optional): Epsilon for stability. Defaults to 1e-8.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, predicted, target_mask):
        """Calculates Focal Loss between prediction and target.

        Args:
            predicted (torch.Tensor): Predicted probabilities.
            target_mask (torch.Tensor): Binary target mask.

        Returns:
            torch.Tensor: Loss tensor.
        """
        pt = torch.where(target_mask, predicted, 1 - predicted)
        pt = torch.clamp(pt, min=self.epsilon, max=1-self.epsilon)
        loss = - self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        return loss


class SegLoss(nn.Module):
    """Calculates segmentation loss using Focal Loss.

    This loss compares predicted segmentation masks with ground truth masks.
    """
    def __init__(self):
        """Initializes SegLoss."""
        super().__init__()

    def forward(self, out_seg, seg):
        """Calculates the segmentation loss.

        Args:
            out_seg (torch.Tensor): Predicted segmentation masks (Batch, SegNum, H, W).
            seg (torch.Tensor): Ground truth segmentation masks.

        Returns:
            torch.Tensor: Scalar loss value.
        """
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
