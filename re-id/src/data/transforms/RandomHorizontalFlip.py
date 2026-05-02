import torch
from torch import nn
from torchvision.transforms import functional as F


def should_horizontal_flip(p=0.5, generator=None):
    """Determines whether a horizontal flip should be performed.

    Args:
        p (float, optional): Probability of flipping. Defaults to 0.5.
        generator (torch.Generator, optional): Random number generator. Defaults to None.

    Returns:
        bool: True if flip should occur, False otherwise.
    """
    if generator:
        chance = torch.rand(1, generator=generator, device=generator.device)
    else:
        chance = torch.rand(1)

    return chance < p


class RandomHorizontalFlip(nn.Module):
    """Randomly flips an image horizontally.

    Attributes:
        p (float): Probability of flipping.
        generator (torch.Generator, optional): Random number generator.
    """
    def __init__(self, p=0.5, generator=None):
        """Initializes RandomHorizontalFlip.

        Args:
            p (float, optional): Probability. Defaults to 0.5.
            generator (torch.Generator, optional): Random generator. Defaults to None.
        """
        super().__init__()
        self.p = p
        self.generator = generator

    def forward(self, img):
        """Applies random horizontal flip to an image.

        Args:
            img (torch.Tensor or PIL.Image): Input image.

        Returns:
            torch.Tensor or PIL.Image: Flipped or original image.
        """
        do_flip = should_horizontal_flip(p=self.p, generator=self.generator)

        if do_flip:
            return F.hflip(img)
        return img