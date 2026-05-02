import torch
import torch.nn as nn
from functools import partial
from typing import Callable, List, Optional
from src.utils.file_path import get_pretrained_file_path
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensures that all layers have a channel number that is divisible by 8.

    Args:
        v (float): Original channel value.
        divisor (int): The divisor to make the value divisible by.
        min_value (int, optional): Minimum value for the result. Defaults to None.

    Returns:
        int: The adjusted channel value.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    """Inverted Residual block as described in MobileNetV3.

    Attributes:
        use_res_connect (bool): Whether to use a residual connection.
        block (nn.Sequential): The sequence of layers in the block.
        out_channels (int): Number of output channels.
        _is_cn (bool): Whether this block is a channel-increasing block.
    """
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        use_hs: bool,
        stride: int,
        dilation: int,
        norm_layer: Optional[Callable[..., nn.Module]],
        se_layer = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        """Initializes the InvertedResidual block.

        Args:
            input_channels (int): Number of input channels.
            kernel (int): Kernel size for depthwise convolution.
            expanded_channels (int): Number of channels after expansion.
            out_channels (int): Number of output channels.
            use_se (bool): Whether to use Squeeze-and-Excitation.
            use_hs (bool): Whether to use Hard-Swish activation.
            stride (int): Stride for the depthwise convolution.
            dilation (int): Dilation for the depthwise convolution.
            norm_layer (Callable, optional): Normalization layer.
            se_layer (Callable, optional): Squeeze-and-Excitation layer.
        """
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if use_hs else nn.ReLU

        # expand
        if expanded_channels != input_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if dilation > 1 else stride
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel,
                stride=stride,
                dilation=dilation,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if use_se:
            squeeze_channels = _make_divisible(expanded_channels // 4, 8)
            layers.append(se_layer(expanded_channels, squeeze_channels))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels
        self._is_cn = stride > 1

    def forward(self, input):
        """Forward pass of the InvertedResidual block.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
class Features(nn.Sequential):
    """MobileNetV3 Feature extractor.

    Attributes:
        model_name (str): Name of the model ('mobilenetv3_small' or 'mobilenetv3_large').
        use_fpn (bool): Whether to prepare for FPN output.
        irb_configs (list[dict]): Configuration for each Inverted Residual block.
        last_ch_out (int): Number of channels in the final feature map.
        fpn_factors (list): Factors for FPN construction.
        downsample_ratio (int): Cumulative downsampling ratio.
    """
    def __init__(self, variant="small", pretrained=False, use_fpn=False):
        """Initializes the Features sequence.

        Args:
            variant (str, optional): Model variant ('small' or 'large'). Defaults to "small".
            pretrained (bool, optional): Whether to load pretrained weights. Defaults to False.
            use_fpn (bool, optional): Whether to output multiple feature maps for FPN. Defaults to False.

        Raises:
            Exception: If variant is invalid.
        """
        super().__init__()
        self.model_name = f"mobilenetv3_{variant}"
        self.use_fpn = use_fpn

        # Inverted Residual Blocks
        if variant == "large":
            self.irb_configs = [
                {"input_ch": 16,  "kernel": 3, "expanded_ch": 16,  "out_ch": 16,  "use_se": False, "use_hs": False, "stride": 1, "dilation": 1, "fpn_layer": True},
                {"input_ch": 16,  "kernel": 3, "expanded_ch": 64,  "out_ch": 24,  "use_se": False, "use_hs": False, "stride": 2, "dilation": 1, "fpn_layer": False},
                {"input_ch": 24,  "kernel": 3, "expanded_ch": 72,  "out_ch": 24,  "use_se": False, "use_hs": False, "stride": 1, "dilation": 1, "fpn_layer": True},
                {"input_ch": 24,  "kernel": 5, "expanded_ch": 72,  "out_ch": 40,  "use_se": True,  "use_hs": False, "stride": 2, "dilation": 1, "fpn_layer": False},
                {"input_ch": 40,  "kernel": 5, "expanded_ch": 120, "out_ch": 40,  "use_se": True,  "use_hs": False, "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 40,  "kernel": 5, "expanded_ch": 120, "out_ch": 40,  "use_se": True,  "use_hs": False, "stride": 1, "dilation": 1, "fpn_layer": True},
                {"input_ch": 40,  "kernel": 3, "expanded_ch": 240, "out_ch": 80,  "use_se": False, "use_hs": True,  "stride": 2, "dilation": 1, "fpn_layer": False},
                {"input_ch": 80,  "kernel": 3, "expanded_ch": 200, "out_ch": 80,  "use_se": False, "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 80,  "kernel": 3, "expanded_ch": 184, "out_ch": 80,  "use_se": False, "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 80,  "kernel": 3, "expanded_ch": 184, "out_ch": 80,  "use_se": False, "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 80,  "kernel": 3, "expanded_ch": 480, "out_ch": 112, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 112, "kernel": 3, "expanded_ch": 672, "out_ch": 112, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 112, "kernel": 5, "expanded_ch": 672, "out_ch": 160, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 160, "kernel": 5, "expanded_ch": 960, "out_ch": 160, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 160, "kernel": 5, "expanded_ch": 960, "out_ch": 160, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False}
            ]
        elif variant == "small":
            self.irb_configs = [
                {"input_ch": 16, "kernel": 3, "expanded_ch": 16,  "out_ch": 16, "use_se": False,  "use_hs": False, "stride": 2, "dilation": 1, "fpn_layer": True},
                {"input_ch": 16, "kernel": 3, "expanded_ch": 72,  "out_ch": 24, "use_se": False, "use_hs": False, "stride": 2, "dilation": 1, "fpn_layer": False},
                {"input_ch": 24, "kernel": 3, "expanded_ch": 88,  "out_ch": 24, "use_se": False, "use_hs": False, "stride": 1, "dilation": 1, "fpn_layer": True},
                {"input_ch": 24, "kernel": 5, "expanded_ch": 96,  "out_ch": 40, "use_se": True,  "use_hs": True,  "stride": 2, "dilation": 1, "fpn_layer": False},
                {"input_ch": 40, "kernel": 5, "expanded_ch": 240, "out_ch": 40, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 40, "kernel": 5, "expanded_ch": 240, "out_ch": 40, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 40, "kernel": 5, "expanded_ch": 120, "out_ch": 48, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 48, "kernel": 5, "expanded_ch": 144, "out_ch": 48, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": True},
                {"input_ch": 48, "kernel": 5, "expanded_ch": 288, "out_ch": 96, "use_se": True,  "use_hs": True,  "stride": 2, "dilation": 1, "fpn_layer": False},
                {"input_ch": 96, "kernel": 5, "expanded_ch": 576, "out_ch": 96, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False},
                {"input_ch": 96, "kernel": 5, "expanded_ch": 576, "out_ch": 96, "use_se": True,  "use_hs": True,  "stride": 1, "dilation": 1, "fpn_layer": False}
            ]
            
        else:
            raise Exception(f"Invalid variant {variant}")

        self.last_ch_out = self.irb_configs[-1]["out_ch"] * 6

        if self.use_fpn:
            self.fpn_factors = []
            cum_downsample = 1
            for cfg in self.irb_configs:
                cum_downsample *= cfg["stride"]
                if cfg["fpn_layer"]:
                    self.fpn_factors.append((cfg["out_ch"], cum_downsample))
            self.fpn_factors.append((self.last_ch_out, cum_downsample))

        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []

        layers.append(
            Conv2dNormActivation(
                3,
                self.irb_configs[0]["input_ch"],
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish
            )
        )

        self.downsample_ratio = 2

        for cfg in self.irb_configs:
            self.downsample_ratio *= cfg["stride"]
            if use_fpn and cfg["fpn_layer"]:
                break

        for cfg in self.irb_configs:
            layers.append(
                InvertedResidual(
                    cfg["input_ch"],
                    cfg["kernel"],
                    cfg["expanded_ch"],
                    cfg["out_ch"],
                    cfg["use_se"],
                    cfg["use_hs"],
                    cfg["stride"],
                    cfg["dilation"],
                    norm_layer
                )
            )

        layers.append(
            Conv2dNormActivation(
                self.irb_configs[-1]["out_ch"],
                self.last_ch_out,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish
            )
        )

        super().__init__(*layers)

        if pretrained:
            try:
                self.load_state_dict(torch.load(get_pretrained_file_path(f"{self.model_name}.pt")))
            except:
                print(f"Failed to load pretrained weight")

    def forward(self, x):
        """Forward pass of the Features sequence.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor or list[torch.Tensor]: Final feature map, or list of feature maps if use_fpn is True.
        """
        if self.use_fpn:
            fpn_outs = []
            for i, module in enumerate(self):
                x = module(x)
                if i > 0 and i <= len(self.irb_configs):
                    if self.irb_configs[i-1]["fpn_layer"]:
                        fpn_outs.append(x)
                elif i == len(self.irb_configs) + 1:
                    fpn_outs.append(x)
            return fpn_outs
        else:
            return super().forward(x)


class FPN(nn.Module):
    """Feature Pyramid Network (FPN) implementation.

    Attributes:
        local_feat_layers (int): Number of lower layers to use for local features.
        out_ch (int): Number of channels in the output feature maps.
        local_conv1x1 (nn.ModuleList): 1x1 convolutions for local feature branches.
        local_conv3x3 (nn.ModuleList): 3x3 convolutions for local feature fusion.
        local_upsamples (list): Upsampling layers for local feature branches.
        fused_conv1x1 (nn.ModuleList): 1x1 convolutions for all feature branches.
        fused_conv3x3 (nn.ModuleList): 3x3 convolutions for global feature fusion.
        fused_upsamples (list): Upsampling layers for global feature branches.
    """
    def __init__(self, fpn_factors, out_ch=None, local_feat_layers=2):
        """Initializes the FPN module.

        Args:
            fpn_factors (list): List of (channels, downsample_factor) for each input feature map.
            out_ch (int, optional): Number of output channels. Defaults to the minimum input channels.
            local_feat_layers (int, optional): Number of layers to use for local features. Defaults to 2.
        """
        super().__init__()
        self.local_feat_layers = local_feat_layers

        if out_ch is None:
            for f in fpn_factors:
                if out_ch is None:
                    out_ch = f[0]
                else:
                    out_ch = out_ch if out_ch <= f[0] else f[0]

        self.out_ch = out_ch

        self.local_conv1x1 = nn.ModuleList()
        self.local_upsamples = []
        self.local_conv3x3 = nn.ModuleList()

        for i in range(local_feat_layers):
            f = fpn_factors[i]
            self.local_conv1x1.append(Conv2dNormActivation(f[0],
                                                           out_ch,
                                                           kernel_size=1,
                                                           stride=1,
                                                           activation_layer=nn.Hardswish))

        for i in range(local_feat_layers - 1, 0, -1):
            h = fpn_factors[i]
            l = fpn_factors[i-1]
            self.local_conv3x3.append(Conv2dNormActivation(out_ch,
                                                           out_ch,
                                                           kernel_size=3,
                                                           stride=1,
                                                           activation_layer=nn.Hardswish))
            scale_factor = h[1] // l[1]
            self.local_upsamples.append(nn.Upsample(scale_factor=scale_factor, mode="nearest"))

        self.fused_conv1x1 = nn.ModuleList()
        self.fused_upsamples = []
        self.fused_conv3x3 = nn.ModuleList()

        for f in fpn_factors:
            self.fused_conv1x1.append(Conv2dNormActivation(f[0],
                                                           out_ch,
                                                           kernel_size=1,
                                                           stride=1,
                                                           activation_layer=nn.Hardswish))
        for i in range(len(fpn_factors) - 1, 0, -1):
            h = fpn_factors[i]
            l = fpn_factors[i-1]
            self.fused_conv3x3.append(Conv2dNormActivation(out_ch,
                                                           out_ch,
                                                           kernel_size=3,
                                                           stride=1,
                                                           activation_layer=nn.Hardswish))
            scale_factor = h[1] // l[1]
            self.fused_upsamples.append(nn.Upsample(scale_factor=scale_factor, mode="nearest"))

    def forward(self, fpn_input):
        """Forward pass of the FPN.

        Args:
            fpn_input (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (fused_feat, local_feat)
                fused_feat: Feature map fused from all input levels.
                local_feat: Feature map fused from lower input levels.
        """
        # Local Feature

        local_reduced = []

        for i in range(self.local_feat_layers):
            local_reduced.append(self.local_conv1x1[i](fpn_input[i]))
        
        local_feat = local_reduced.pop()

        for i in range(self.local_feat_layers - 1):
            h = local_feat
            l = local_reduced.pop()

            upsample = self.local_upsamples[i]
            conv3x3 = self.local_conv3x3[i]

            local_feat = conv3x3(torch.add(upsample(h), l))

        # Fused Feature

        fused_reduced = []

        for i in range(len(fpn_input)):
            fused_reduced.append(self.fused_conv1x1[i](fpn_input[i]))

        fused_feat = fused_reduced.pop()

        for i in range(len(self.fused_upsamples)):
            h = fused_feat
            l = fused_reduced.pop()

            upsample = self.fused_upsamples[i]
            conv3x3 = self.fused_conv3x3[i]

            fused_feat = conv3x3(torch.add(upsample(h), l))

        return fused_feat, local_feat

class MobilenetV3(nn.Module):
    """MobileNetV3 model with optional FPN.

    Attributes:
        features (Features): Feature extraction sequence.
        use_fpn (bool): Whether to use FPN.
        downsample_ratio (int): Cumulative downsampling ratio.
        fpn (FPN, optional): Feature Pyramid Network module.
        out_ch (int): Number of channels in the final output feature map.
    """
    def __init__(self, variant, pretrained=False, use_fpn=False, fpn_out_ch=None):
        """Initializes the MobilenetV3 model.

        Args:
            variant (str): Model variant ('small' or 'large').
            pretrained (bool, optional): Whether to load pretrained weights. Defaults to False.
            use_fpn (bool, optional): Whether to use FPN. Defaults to False.
            fpn_out_ch (int, optional): Number of FPN output channels. Defaults to None.
        """
        super().__init__()
        self.features = Features(variant=variant, pretrained=pretrained, use_fpn=use_fpn)
        self.use_fpn = use_fpn
        self.downsample_ratio = self.features.downsample_ratio
        if use_fpn:
            self.fpn = FPN(self.features.fpn_factors, out_ch=fpn_out_ch)
            self.out_ch = self.fpn.out_ch
        else:
            self.out_ch = self.features.last_ch_out

    def forward(self, x):
        """Forward pass of the MobilenetV3 model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]: Output feature map(s).
        """
        x = self.features(x)
        if self.use_fpn:
            x = self.fpn(x)
        return x
