import torch
from torch import nn
from torchvision.transforms import functional as F


def get_random_shift_params(p=1, padding=10, generator=None):
    if generator:
        chance = torch.rand(1, generator=generator, device=generator.device)
    else:
        chance = torch.rand(1)

    if chance >= p:
        return False, None

    if generator:
        x_shift = torch.randint(
            0, padding * 2 + 1, [1], generator=generator, device=generator.device).item()
        y_shift = torch.randint(
            0, padding * 2 + 1, [1], generator=generator, device=generator.device).item()
    else:
        x_shift = torch.randint(0, padding * 2 + 1, [1]).item()
        y_shift = torch.randint(0, padding * 2 + 1, [1]).item()

    return True, (padding, x_shift, y_shift)


def shift_image(img: torch.Tensor, padding, x_shift, y_shift):
    height = img.size(1)
    width = img.size(2)

    img = F.pad(img, [padding, padding, padding, padding], fill=0)
    img = F.crop(img, y_shift, x_shift, height, width)

    return img


class RandomShift(nn.Module):
    def __init__(self, p=1, padding=10, generator=None):
        super().__init__()
        self.p = p
        self.padding = padding
        self.generator = generator

    def forward(self, img):
        do_shift, params = get_random_shift_params(
            self.p, self.padding, self.generator)

        if not do_shift:
            return img

        padding, x_shift, y_shift = params
        return shift_image(img, padding, x_shift, y_shift)
