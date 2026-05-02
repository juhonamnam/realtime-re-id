import math
import torch
from torch import nn


class RandomErasing(nn.Module):
    """Randomly erases a rectangular region in an image tensor.

    Attributes:
        p (float): Probability of applying erasing.
        scale (tuple): Range of proportion of erased area.
        ratio (tuple): Range of aspect ratio of erased area.
        value (float or list, optional): Value to fill the erased area.
        generator (torch.Generator, optional): Random number generator.
    """
    def __init__(self, p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=None, generator=None):
        """Initializes RandomErasing.

        Args:
            p (float, optional): Probability. Defaults to 0.3.
            scale (tuple, optional): Scale range. Defaults to (0.02, 0.33).
            ratio (tuple, optional): Aspect ratio range. Defaults to (0.3, 3.3).
            value (float or list, optional): Erasing value. Defaults to None (random).
            generator (torch.Generator, optional): Random generator. Defaults to None.
        """
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.generator = generator

    def forward(self, img):
        """Applies random erasing to an image tensor.

        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Image tensor with random erasing applied.
        """
        if self.generator:
            p = torch.rand(1, generator=self.generator, device=self.generator.device)
        else:
            p = torch.rand(1)

        if p > self.p:
            return img

        area = img.size()[1] * img.size()[2]

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

        h = h if h < img.size()[1] else img.size()[1]
        w = w if w < img.size()[2] else img.size()[2]

        if self.generator:
            x1 = torch.randint(0, img.size()[1] - h + 1, [1], generator=self.generator, device=self.generator.device).item()
            y1 = torch.randint(0, img.size()[2] - w + 1, [1], generator=self.generator, device=self.generator.device).item()
        else:
            x1 = torch.randint(0, img.size()[1] - h + 1, [1]).item()
            y1 = torch.randint(0, img.size()[2] - w + 1, [1]).item()

        if self.value is not None:
            if isinstance(self.value, list) or isinstance(self.value, tuple):
                value = self.value
            else:
                value = [self.value, self.value, self.value]
        else:
            if self.generator:
                value = torch.rand(3, generator=self.generator, device=self.generator.device).tolist()
            else:
                value = torch.rand(3).tolist()

        img[0, x1:x1 + h, y1:y1 + w] = value[0]
        img[1, x1:x1 + h, y1:y1 + w] = value[1]
        img[2, x1:x1 + h, y1:y1 + w] = value[2]
        return img