import math
import torch
from torch import nn
from torchvision.transforms import functional as F


class RandomCrop(nn.Module):
    def __init__(self, p=0.3, scale=(0.5, 1), ratio=(0.2, 1), generator=None):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.generator = generator

    def forward(self, img):

        if self.generator:
            p = torch.rand(1, generator=self.generator, device=self.generator.device)
        else:
            p = torch.rand(1)

        if p > self.p:
            return img

        if isinstance(img, torch.Tensor):
            resolution = (img.size()[2], img.size()[1])
        else:
            resolution = (img.size[0], img.size[1])

        (width, height) = resolution

        area = width * height

        if self.generator:
            rn = torch.rand(2, generator=self.generator, device=self.generator.device)
        else:
            rn = torch.rand(2)

        target_area = rn[0] * (self.scale[1] - self.scale[0]) + self.scale[0]
        target_area *= area

        r1 = self.ratio[0]
        r2 = self.ratio[1]
        if r1 > 1:
            r1 = 2 - 1 / r1
        if r2 > 1:
            r2 = 2 - 1 / r2
        
        aspect_ratio = rn[1] * (r2 - r1) + r1
        if aspect_ratio > 1:
            aspect_ratio = 1 / (2 - aspect_ratio)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        h = h if h < height else height
        w = w if w < width else width

        if self.generator:
            x1 = torch.randint(0, width - w + 1, [1], generator=self.generator, device=self.generator.device).item()
            y1 = torch.randint(0, height - h + 1, [1], generator=self.generator, device=self.generator.device).item()
        else:
            x1 = torch.randint(0, width - w + 1, [1]).item()
            y1 = torch.randint(0, height - h + 1, [1]).item()

        return F.crop(img, y1, x1, h, w)
