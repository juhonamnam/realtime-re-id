import torch
from torch import nn
from torchvision.transforms import functional as F


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5, generator=None):
        super().__init__()
        self.p = p
        self.generator = generator

    def forward(self, img):
        if self.generator:
            p = torch.rand(1, generator=self.generator, device=self.generator.device)
        else:
            p = torch.rand(1)

        if p > self.p:
            return F.hflip(img)
        return img