import math
import torch
from torch import nn
from torchvision.transforms import functional as F


def get_random_crop_params(resolution, p=0.3, scale=(0.5, 1), ratio=(0.2, 1), generator=None):
    """Calculates parameters for a random crop.

    Args:
        resolution (tuple[int, int]): Input resolution (Height, Width).
        p (float, optional): Probability of applying the crop. Defaults to 0.3.
        scale (tuple[float, float], optional): Range of scale for the crop area. Defaults to (0.5, 1).
        ratio (tuple[float, float], optional): Range of aspect ratio for the crop. Defaults to (0.2, 1).
        generator (torch.Generator, optional): Random number generator. Defaults to None.

    Returns:
        tuple: (do_crop, params)
            do_crop (bool): Whether to perform the crop.
            params (tuple[int, int, int, int]): (y1, x1, h, w) of the crop if do_crop is True.
    """
    if generator:
        chance = torch.rand(1, generator=generator, device=generator.device)
    else:
        chance = torch.rand(1)

    if chance >= p:
        return False, None

    (height, width) = resolution

    area = width * height

    if generator:
        rn = torch.rand(2, generator=generator, device=generator.device)
    else:
        rn = torch.rand(2)

    target_area = rn[0] * (scale[1] - scale[0]) + scale[0]
    target_area *= area

    r1 = ratio[0]
    r2 = ratio[1]
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

    if generator:
        x1 = torch.randint(0, width - w + 1, [1], generator=generator, device=generator.device).item()
        y1 = torch.randint(0, height - h + 1, [1], generator=generator, device=generator.device).item()
    else:
        x1 = torch.randint(0, width - w + 1, [1]).item()
        y1 = torch.randint(0, height - h + 1, [1]).item()

    return True, (y1, x1, h, w)


class RandomCrop(nn.Module):
    """Randomly crops an image based on scale and aspect ratio.

    Attributes:
        p (float): Probability of applying the crop.
        scale (tuple): Range of scale for the crop area.
        ratio (tuple): Range of aspect ratio.
        generator (torch.Generator, optional): Random number generator.
    """
    def __init__(self, p=0.3, scale=(0.5, 1), ratio=(0.2, 1), generator=None):
        """Initializes RandomCrop.

        Args:
            p (float, optional): Probability. Defaults to 0.3.
            scale (tuple, optional): Area scale range. Defaults to (0.5, 1).
            ratio (tuple, optional): Aspect ratio range. Defaults to (0.2, 1).
            generator (torch.Generator, optional): Random generator. Defaults to None.
        """
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.generator = generator

    def forward(self, img):
        """Applies the random crop to an image.

        Args:
            img (torch.Tensor or PIL.Image): Input image.

        Returns:
            torch.Tensor or PIL.Image: Cropped image.
        """

        if isinstance(img, torch.Tensor):
            resolution = (img.size()[1], img.size()[2])
        else:
            resolution = (img.size[1], img.size[0])

        do_crop, params = get_random_crop_params(resolution, self.p, self.scale, self.ratio, self.generator)

        if not do_crop:
            return img

        y1, x1, h, w = params
        return F.crop(img, y1, x1, h, w)
