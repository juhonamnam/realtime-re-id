import torch
from torch import nn
from torchvision.transforms import functional as F


class RandomResolutionReduce(nn.Module):
    def __init__(self, target_size, p=0.3, min_ratio=0.25,
                 interpolation=F.InterpolationMode.NEAREST,
                 generator=None):
        super().__init__()
        self.target_size = target_size
        self.p = p
        self.min_ratio = min_ratio
        self.interpolation = interpolation
        self.generator = generator

    def forward(self, img):
        if self.generator:
            chance = torch.rand(1, generator=self.generator, device=self.generator.device)
        else:
            chance = torch.rand(1)

        if chance >= self.p:
            return img

        if self.generator:
            reduction_factor = torch.rand(1, generator=self.generator, device=self.generator.device)
        else:
            reduction_factor = torch.rand(1)

        reduction_factor = self.min_ratio + (1 - self.min_ratio) * reduction_factor

        new_resolution = (int(self.target_size[0] * reduction_factor),
                          int(self.target_size[1] * reduction_factor))

        img = F.resize(img, new_resolution, self.interpolation)
        img = F.resize(img, self.target_size, self.interpolation)

        return img