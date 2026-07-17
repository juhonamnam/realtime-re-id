import math
import torch
from torch import nn


def get_random_erasing_params(height, width, p=0.5, sl=0.02, sh=0.4, r1=0.3, generator=None):
    if generator:
        chance = torch.rand(1, generator=generator, device=generator.device)
    else:
        chance = torch.rand(1)

    if chance >= p:
        return False, None

    for attempt in range(100):
        area = width * height

        if generator:
            target_area = torch.empty(1, device=generator.device).uniform_(
                sl, sh, generator=generator).item() * area
            aspect_ratio = torch.empty(1, device=generator.device).uniform_(
                r1, 1 / r1, generator=generator).item()
        else:
            target_area = torch.empty(1).uniform_(
                sl, sh).item() * area
            aspect_ratio = torch.empty(1).uniform_(
                r1, 1 / r1).item()

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < width and h < height:

            if generator:
                x1 = torch.randint(
                    0, height - h + 1, [1], generator=generator, device=generator.device).item()
                y1 = torch.randint(
                    0, width - w + 1, [1], generator=generator, device=generator.device).item()
            else:
                x1 = torch.randint(0, height - h + 1, [1]).item()
                y1 = torch.randint(0, width - w + 1, [1]).item()

            return True, (x1, y1, h, w)

    return False, None


def fill_rectangle(img, x1, y1, h, w, fill_value=[0.4914, 0.4822, 0.4465]):
    if isinstance(fill_value, (list, tuple)):
        for c in range(img.size(0)):
            img[c, x1:x1 + h, y1:y1 + w] = fill_value[c]
    else:
        img[:, x1:x1 + h, y1:y1 + w] = fill_value

    return img


class RandomErasing(nn.Module):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, fill_value=[0.4914, 0.4822, 0.4465], generator=None):
        super().__init__()
        self.p = p
        self.fill_value = fill_value
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.generator = generator

    def forward(self, img):
        do_erase, params = get_random_erasing_params(img.size()[1], img.size(
        )[2], p=self.p, sl=self.sl, sh=self.sh, r1=self.r1, generator=self.generator)

        if not do_erase:
            return img

        x1, y1, h, w = params

        return fill_rectangle(img, x1, y1, h, w, self.fill_value)
