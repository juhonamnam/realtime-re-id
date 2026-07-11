import torch
from torch import nn
from torchvision.transforms import functional as F


def get_random_resolution_reduce_params(target_size, p=0.3, min_ratio=0.25, generator=None):
    if generator:
        chance = torch.rand(1, generator=generator, device=generator.device)
    else:
        chance = torch.rand(1)

    if chance >= p:
        return False, None

    if generator:
        reduction_factor = torch.rand(
            1, generator=generator, device=generator.device)
    else:
        reduction_factor = torch.rand(1)

    reduction_factor = min_ratio + (1 - min_ratio) * reduction_factor

    new_resolution = (int(target_size[0] * reduction_factor),
                      int(target_size[1] * reduction_factor))

    return True, new_resolution


def do_redolution_reduce(img, target_size, new_resolution):
    img = F.resize(img, new_resolution, F.InterpolationMode.NEAREST)
    img = F.resize(img, target_size, F.InterpolationMode.NEAREST)
    return img


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
        do_reduce, new_resolution = get_random_resolution_reduce_params(
            self.target_size, self.p, self.min_ratio, self.generator)

        if not do_reduce:
            return img

        return do_redolution_reduce(img, self.target_size, new_resolution)
