import torch
from torch import nn
from torchvision.transforms import functional as F


def should_horizontal_flip(p=0.5, generator=None):
    if generator:
        chance = torch.rand(1, generator=generator, device=generator.device)
    else:
        chance = torch.rand(1)

    return chance < p


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5, generator=None):
        super().__init__()
        self.p = p
        self.generator = generator

    def forward(self, img):
        do_flip = should_horizontal_flip(p=self.p, generator=self.generator)

        if do_flip:
            return F.hflip(img)
        return img