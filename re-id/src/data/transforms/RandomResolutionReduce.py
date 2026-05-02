import torch
from torch import nn
from torchvision.transforms import functional as F


class RandomResolutionReduce(nn.Module):
    """Randomly reduces image resolution and then scales it back to target size.

    This simulates low-resolution inputs as a form of data augmentation.

    Attributes:
        target_size (tuple): The final resolution (Height, Width).
        p (float): Probability of applying the resolution reduction.
        min_ratio (float): Minimum reduction factor for the resolution.
        interpolation (F.InterpolationMode): Interpolation method used.
        generator (torch.Generator, optional): Random number generator.
    """
    def __init__(self, target_size, p=0.3, min_ratio=0.25,
                 interpolation=F.InterpolationMode.NEAREST,
                 generator=None):
        """Initializes RandomResolutionReduce.

        Args:
            target_size (tuple): Target (Height, Width).
            p (float, optional): Probability. Defaults to 0.3.
            min_ratio (float, optional): Min scale ratio. Defaults to 0.25.
            interpolation (F.InterpolationMode, optional): Interpolation method. Defaults to NEAREST.
            generator (torch.Generator, optional): Random generator. Defaults to None.
        """
        super().__init__()
        self.target_size = target_size
        self.p = p
        self.min_ratio = min_ratio
        self.interpolation = interpolation
        self.generator = generator

    def forward(self, img):
        """Applies random resolution reduction to an image.

        Args:
            img (torch.Tensor or PIL.Image): Input image.

        Returns:
            torch.Tensor or PIL.Image: Transformed image.
        """
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