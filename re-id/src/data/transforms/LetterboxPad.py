import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


def get_letterbox_pad_params(image, target_size, random_fill=True, p=(0.3, 0.3), generator=None):
    target_height, target_width = target_size

    if random_fill:
        full_padding_prob, no_padding_prob = p
        if generator:
            chance = torch.rand(1, generator=generator,
                                device=generator.device)
        else:
            chance = torch.rand(1)

        if chance < full_padding_prob:
            random_fill = False
        elif chance < full_padding_prob + no_padding_prob:
            # No padding, fill entire target size
            return target_height, target_width, 0, 0, 0, 0

    if image.dim() != 3:
        raise ValueError(
            f"Expected 3D tensor (C, H, W), got {image.dim()}D tensor")

    _, H, W = image.shape

    # Calculate scale factor to fit image in target size while maintaining aspect ratio
    scale_h = target_height / H
    scale_w = target_width / W
    scale = min(scale_h, scale_w)  # Use smaller scale to ensure image fits

    # Calculate new dimensions
    new_h = int(H * scale)
    new_w = int(W * scale)

    if random_fill:
        # Randomly choose new dimensions within the target size
        if generator:
            new_h = torch.randint(new_h, target_height + 1, (1,),
                                  generator=generator, device=generator.device).item()
            new_w = torch.randint(new_w, target_width + 1, (1,),
                                  generator=generator, device=generator.device).item()
        else:
            new_h = torch.randint(new_h, target_height + 1, (1,)).item()
            new_w = torch.randint(new_w, target_width + 1, (1,)).item()

    # Calculate padding needed
    pad_h = target_height - new_h
    pad_w = target_width - new_w

    # Calculate padding for each side (distribute evenly)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return new_h, new_w, pad_left, pad_right, pad_top, pad_bottom


def apply_letterbox_pad(image, new_h, new_w, pad_left, pad_right,
                        pad_top, pad_bottom, pad_value=0,
                        interpolation=transforms.InterpolationMode.NEAREST):
    # Resize image
    resized = TF.resize(
        image,
        size=(new_h, new_w),
        interpolation=interpolation,
        antialias=True
    )

    resized = torch.clamp(resized, 0, 1)

    # Apply padding: F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
    padded = F.pad(
        resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=pad_value
    )

    return padded


class LetterboxPad(nn.Module):

    def __init__(self, size, pad_value=0, random_fill=True, p=(0.3, 0.3),
                 interpolation=transforms.InterpolationMode.NEAREST, generator=None):
        super(LetterboxPad, self).__init__()
        self.size = size
        self.pad_value = pad_value
        self.random_fill = random_fill
        self.p = p
        self.interpolation = interpolation
        self.generator = generator

    def forward(self, image):
        letterbox_params = get_letterbox_pad_params(
            image, self.size, random_fill=self.random_fill, generator=self.generator)
        return apply_letterbox_pad(image, *letterbox_params,
                                   pad_value=self.pad_value, interpolation=self.interpolation)
