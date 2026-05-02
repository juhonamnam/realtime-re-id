import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

class LetterboxPad(nn.Module):
    """Resizes an image to fit target resolution while maintaining aspect ratio, then pads to exact target size.

    Attributes:
        target_height (int): Target height.
        target_width (int): Target width.
        pad_value (float): Value used for padding.
        interpolation (transforms.InterpolationMode): Interpolation method.
    """
    
    def __init__(self, target_size, pad_value=0, interpolation=transforms.InterpolationMode.NEAREST):
        """Initializes LetterboxPad.

        Args:
            target_size (tuple[int, int]): Target (height, width) resolution.
            pad_value (float, optional): Value to use for padding. Defaults to 0.
            interpolation (transforms.InterpolationMode, optional): Interpolation method. Defaults to NEAREST.
        """
        super(LetterboxPad, self).__init__()
        self.target_height, self.target_width = target_size
        self.pad_value = pad_value
        self.interpolation = interpolation

    def forward(self, image):
        """Applies the letterbox padding transform to an image tensor.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).
        
        Returns:
            torch.Tensor: Transformed image of shape (C, target_height, target_width).

        Raises:
            ValueError: If input tensor is not 3D.
        """
        if image.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got {image.dim()}D tensor")
        
        C, H, W = image.shape
        
        # Calculate scale factor to fit image in target size while maintaining aspect ratio
        scale_h = self.target_height / H
        scale_w = self.target_width / W
        scale = min(scale_h, scale_w)  # Use smaller scale to ensure image fits
        
        # Calculate new dimensions
        new_h = int(H * scale)
        new_w = int(W * scale)

        # Resize image
        resized = TF.resize(
            image,
            size=(new_h, new_w),
            interpolation=self.interpolation,
            antialias=True
        )

        resized = torch.clamp(resized, 0, 1)

        # Calculate padding needed
        pad_h = self.target_height - new_h
        pad_w = self.target_width - new_w
        
        # Calculate padding for each side (distribute evenly)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Apply padding: F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        padded = F.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=self.pad_value
        )
        
        return padded
    
    def __repr__(self):
        return f"{self.__class__.__name__}(target_size=({self.target_height}, {self.target_width}), pad_value={self.pad_value})"