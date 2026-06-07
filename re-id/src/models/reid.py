import torch
import torch.nn as nn
import torch.nn.functional as F
from src.metric import get_emb_similarity_vector

from torchvision.ops.misc import Conv2dNormActivation

from src.utils.segment_info import get_segment_groups, get_attention_groups
from .mobilenetv3 import MobilenetV3
from src.utils.file_path import get_weight_file_path, get_pretrained_file_path, get_export_file_path

__all__ = ['ReIDModel']


# ImageNet Normalization
class Normalize(nn.Module):
    """Normalizes input tensors using ImageNet mean and standard deviation.

    Attributes:
        mean (torch.Tensor): ImageNet mean values for RGB channels.
        std (torch.Tensor): ImageNet standard deviation values for RGB channels.
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Initializes the Normalize module.

        Args:
            mean (list[float], optional): Mean values for normalization. Defaults to [0.485, 0.456, 0.406].
            std (list[float], optional): Standard deviation values for normalization. Defaults to [0.229, 0.224, 0.225].
        """
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        """Normalizes the input tensor.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        return (x - self.mean) / self.std

class ReIDModel(nn.Module):
    """Main Person Re-Identification model.

    This model uses a MobileNetV3 backbone with a segmentation-based attention mechanism
    to extract discriminative embeddings for person re-identification.

    Attributes:
        model_variant (str): Variant of the backbone model ('m3small' or 'm3large').
        seg_variant (str): Variant for segmentation logic.
        emb_len (int): Length of the output embedding vector.
        input_resolution (tuple[int, int]): Expected input resolution (Height, Width).
        segment_num (int): Number of segmentation classes.
        attention_groups (list): List of attention group configurations.
        attention_num (int): Number of attention groups.
        seg_to_att_indices (list[int]): Indices mapping segmentation results to attention.
        model_name (str): Unique name identifying the model configuration.
        export_file_path (str): Path to save the PyTorch model file.
        onnx_export_file_path (str): Path to save the exported ONNX model.
        normalize (Normalize): Normalization module.
        backbone (MobilenetV3): Backbone feature extractor.
        gradcam_layer (str): Target layer for GradCAM visualization.
        segmentation (nn.Sequential): Segmentation head.
        emb_vec (nn.ModuleList): Embedding extraction heads for each attention group.
    """
    def __init__(self,
                 model_variant,
                 seg_variant,
                 emb_len,
                 input_resolution,
                 pretrained=False,
                 backbone_pretrained=False):
        """Initializes the ReIDModel.

        Args:
            model_variant (str): Backbone variant ('m3small' or 'm3large').
            seg_variant (str): Segmentation configuration variant.
            emb_len (int): Dimension of the embedding vectors.
            input_resolution (tuple[int, int]): Input (Height, Width).
            pretrained (bool, optional): Whether to load full model weights. Defaults to False.
            backbone_pretrained (bool, optional): Whether to load backbone-only weights. Defaults to False.

        Raises:
            Exception: If model_variant is not supported.
        """
        super().__init__()

        if pretrained:
            backbone_pretrained = False

        self.model_variant = model_variant
        self.seg_variant = seg_variant
        self.emb_len = emb_len
        self.input_resolution = input_resolution

        self.segment_num = len(get_segment_groups(seg_variant))

        self.attention_groups = get_attention_groups(seg_variant)
        self.attention_num = len(self.attention_groups)

        self.seg_to_att_indices = [x["seg_idx"] for x in self.attention_groups]

        self.model_name = self.get_model_name(model_variant, seg_variant, emb_len, input_resolution)

        self.export_file_path = get_export_file_path(f"{self.model_name}.pt")
        self.onnx_export_file_path = get_export_file_path(f"{self.model_name}.onnx")

        self.normalize = Normalize()

        if model_variant == "m3small":
            self.backbone = MobilenetV3(variant="small", pretrained=backbone_pretrained, use_fpn=True, fpn_out_ch=192)
        elif model_variant == "m3large":
            self.backbone = MobilenetV3(variant="large", pretrained=backbone_pretrained, use_fpn=True, fpn_out_ch=192)
        else:
            raise Exception(f"\"{model_variant}\" Not Supported")

        self.gradcam_layer = "backbone.fpn.local_conv3x3.0"

        self.segmentation = nn.Sequential(
            Conv2dNormActivation(self.backbone.out_ch,
                                 96,
                                 kernel_size=3,
                                 stride=1,
                                 activation_layer=nn.Hardswish),
            nn.Conv2d(96, self.segment_num, 1),
            nn.Softmax2d(),
        )

        self.emb_vec_gate = nn.ModuleList([
            nn.Sequential(Conv2dNormActivation(self.backbone.out_ch,
                                               96,
                                               kernel_size=1,
                                               stride=1,
                                               activation_layer=nn.Hardswish),
                          nn.Conv2d(96, self.emb_len, kernel_size=1)) for _ in range(self.attention_num)])

        self.emb_vec = nn.ModuleList([Conv2dNormActivation(self.backbone.out_ch,
                                                           self.emb_len,
                                                           kernel_size=1,
                                                           stride=1,
                                                           activation_layer=nn.Hardswish) for _ in range(self.attention_num)])

        if pretrained:
            try:
                self.load_state_dict(torch.load(get_pretrained_file_path(f"{self.model_name}.pt")))
            except:
                print("Failed to load pretrained weight")

    @staticmethod
    def get_model_name(model_variant, seg_variant, emb_len, input_resolution):
        """Generates a unique model name based on configuration parameters.

        Args:
            model_variant (str): Backbone variant.
            seg_variant (str): Segmentation configuration variant.
            emb_len (int): Embedding vector length.
            input_resolution (tuple[int, int]): Input resolution (Height, Width).

        Returns:
            str: Generated model name.
        """
        model_name = "_".join(["reid",
                               model_variant,
                               f"{seg_variant}{emb_len}e",
                               f"{input_resolution[1]}x{input_resolution[0]}"])
        return model_name

    def _forward(self, x: torch.Tensor):
        """Internal forward pass shared by different model versions.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, 3, H, W).

        Returns:
            tuple: (local_feat, seg, att, v_scores, emb_vecs)
                local_feat (torch.Tensor): High-level feature maps from backbone.
                seg (torch.Tensor): Segmentation maps.
                att (torch.Tensor): Refined attention masks.
                v_scores (torch.Tensor): Visibility scores for each attention group.
                emb_vecs (torch.Tensor): Extracted embedding vectors.
        """
        x = self.normalize(x)
        fused_feat, local_feat = self.backbone(x)                                # batch, out_ch, height, width
        seg = self.segmentation(fused_feat)                                      # batch, segmentation_num, height, width

        seg_max = seg.max(dim=1, keepdim=True)                                   # batch, 1, height, width 

        att = torch.zeros_like(seg)                                              # batch, attention_num, height, width
        att = att.scatter(1, seg_max.indices, seg_max.values)                    # batch, attention_num, height, width

        att = att[:, self.seg_to_att_indices]                                    # batch, attention_num, height, width

        v_scores = att.amax(dim=(2, 3))                                          # batch, attention_num

        emb_vecs = []
        emb_vec_gates = []
        for i in range(self.attention_num):
            emb_vec = self.emb_vec[i](local_feat)                                # batch, emb_len, height, width

            att_weight = att[:, [i]]                                             # batch, 1, height, width

            emb_vec = att_weight * emb_vec                                       # batch, emb_len, height, width
            emb_vec = emb_vec.sum(dim=(2, 3))                                    # batch, emb_len
            emb_vec /= att_weight.sum(dim=(2, 3)).clamp(min=1)                   # batch, emb_len
            emb_vecs.append(emb_vec)

            emb_vec_gate = self.emb_vec_gate[i](fused_feat)                      # batch, emb_len, height, width

            emb_vec_gate = att_weight * emb_vec_gate                             # batch, emb_len, height, width
            emb_vec_gate = emb_vec_gate.sum(dim=(2, 3))                          # batch, emb_len
            emb_vec_gate /= att_weight.sum(dim=(2, 3)).clamp(min=1)              # batch, emb_len
            emb_vec_gate = torch.sigmoid(emb_vec_gate)                           # batch, emb_len
            emb_vec_gates.append(emb_vec_gate)

        emb_vecs = torch.stack(emb_vecs, dim=1)                                  # batch, attention_num, feature_len
        emb_vec_gates = torch.stack(emb_vec_gates, dim=1)                        # batch, attention_num, feature_len

        return local_feat, seg, att, v_scores, emb_vecs, emb_vec_gates

    def forward(self, x: torch.Tensor):
        """Forward pass for the main Re-ID model.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, 3, H, W).

        Returns:
            tuple: ((seg, att), (v_scores, emb_vecs))
                seg (torch.Tensor): Segmentation maps.
                att (torch.Tensor): Refined attention masks.
                v_scores (torch.Tensor): Visibility scores.
                emb_vecs (torch.Tensor): Embedding vectors.
        """
        _, seg, att, v_scores, emb_vecs, emb_vec_gates = self._forward(x)
        return (seg, att), (v_scores, emb_vecs, emb_vec_gates)

    def export(self):
        """Creates an instance of the model optimized for export.

        Returns:
            ReIDExportModel: Model wrapper for ONNX export.
        """
        export_model = ReIDExportModel(self)
        return export_model

    def get_train_model(self, class_num: int):
        """Creates an instance of the model for training.

        Args:
            class_nums (list[int]): Number of classes for the classification head.

        Returns:
            ReIDTrainModel: Model wrapper for training.
        """
        return ReIDTrainModel(self, class_num)

    def get_gradcam_model(self):
        """Creates an instance of the model for GradCAM visualization.

        Returns:
            ReIDGradCAM: Model wrapper with GradCAM hooks.
        """
        return ReIDGradCAM(self)


class ReIDTrainModel(nn.Module):
    """ReIDModel wrapper for training.

    Includes additional heads for classification and utilities for staged training.

    Attributes:
        reid_model (ReIDModel): The base Re-ID model.
        global_features (Conv2dNormActivation): Module to extract global features.
        logit (nn.Linear): Classification head.
        weight_file_path (str): Template path for saving training weights.
    """
    def __init__(self, reid_model: ReIDModel, class_num: int):
        """Initializes the ReIDTrainModel.

        Args:
            reid_model (ReIDModel): Base ReID model instance.
            class_num (int): Number of identities/classes for classification.
        """
        super().__init__()
        self.reid_model = reid_model

        self.global_features = Conv2dNormActivation(self.reid_model.backbone.out_ch,
                                                    self.reid_model.emb_len,
                                                    kernel_size=3,
                                                    stride=1,
                                                    activation_layer=nn.Hardswish)

        self.logit = nn.Linear((self.reid_model.emb_len *
                                (self.reid_model.attention_num + 1)),
                                class_num)

        self.weight_file_path = get_weight_file_path(f"{self.reid_model.model_name}_{class_num}cls_epoch_{{}}.pt")

    def train_backbone(self):
        """Enables gradients for the backbone and segmentation head."""
        for p in self.reid_model.backbone.parameters():
            p.requires_grad = True

        for p in self.reid_model.segmentation.parameters():
            p.requires_grad = True

    def train_segmentation(self):
        """Enables gradients only for the segmentation head, freezes backbone."""
        for p in self.reid_model.backbone.parameters():
            p.requires_grad = False

        for p in self.reid_model.segmentation.parameters():
            p.requires_grad = True

    def train_embvec(self):
        """Freezes both backbone and segmentation head for embedding training."""
        for p in self.reid_model.backbone.parameters():
            p.requires_grad = False

        for p in self.reid_model.segmentation.parameters():
            p.requires_grad = False

    def train_embvec_gate(self):
        """Enables gradients for embedding gates, freezes backbone and segmentation."""
        for p in self.reid_model.backbone.parameters():
            p.requires_grad = False

        for p in self.reid_model.segmentation.parameters():
            p.requires_grad = False

    def forward(self, x):
        """Forward pass for training.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            tuple: (seg, v_scores, emb_vecs, class_logits)
                seg (torch.Tensor): Segmentation maps.
                v_scores (torch.Tensor): Visibility scores.
                emb_vecs (torch.Tensor): Embedding vectors.
                class_logits (torch.Tensor): Predicted classification logits.
        """
        local_feat, seg, att, v_scores, emb_vecs, emb_vec_gates = self.reid_model._forward(x)

        global_feat = self.global_features(local_feat)
        global_att = att.sum(dim=1, keepdim=True)

        global_feat = global_feat * global_att
        global_feat = global_feat.sum(dim=(2, 3))
        global_feat /= global_att.sum(dim=(2, 3)).clamp(min=1e-12)

        ft = emb_vecs
        ft = ft * v_scores.unsqueeze(-1)

        global_v_score = global_att.amax(dim=(2, 3))
        global_feat = global_feat * global_v_score

        concat_feature = torch.cat([global_feat,
                                    ft.flatten(1)],
                                   dim=1)

        class_logits = self.logit(concat_feature)

        return seg, v_scores, emb_vecs, emb_vec_gates, class_logits


class ReIDExportModel(ReIDModel):
    """Simplified version of ReIDModel for ONNX export.

    Only outputs the necessary components (v_scores and emb_vecs).
    """
    def __init__(self, reid_model: ReIDModel):
        """Initializes the export model.

        Args:
            reid_model (ReIDModel): Base model to copy weights from.
        """
        super().__init__(reid_model.model_variant,
                         reid_model.seg_variant,
                         reid_model.emb_len,
                         reid_model.input_resolution)
        self.load_state_dict(reid_model.state_dict())
        self.eval()

    def forward(self, x):
        """Forward pass for exported model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            tuple: (v_scores, emb_vecs)
        """
        _, _, _, v_scores, emb_vecs, emb_vec_gates = super()._forward(x)
        return v_scores, emb_vecs, emb_vec_gates


class ReIDGradCAM(ReIDModel):
    """ReIDModel with GradCAM hooks for feature visualization.

    Attributes:
        hook_registered (bool): Whether forward/backward hooks are registered.
        forward_result (torch.Tensor): Cached output from the target layer.
        backward_result (torch.Tensor): Cached gradient from the target layer.
    """
    def __init__(self, reid_model: ReIDModel):
        """Initializes the GradCAM model.

        Args:
            reid_model (ReIDModel): Base model instance.
        """
        super().__init__(reid_model.model_variant,
                         reid_model.seg_variant,
                         reid_model.emb_len,
                         reid_model.input_resolution)
        self.load_state_dict(reid_model.state_dict())
        self.hook_registered = False
        self.register_hooks()
        
        for p in self.parameters():
            p.requires_grad = True

    def register_hooks(self):
        """Registers forward and backward hooks on the target GradCAM layer."""

        def iterate_module_to_register_hook(module, name=None):
            if name == self.gradcam_layer:
                module.register_forward_hook(self.forward_hook)
                module.register_backward_hook(self.backward_hook)
                self.hook_registered = True
                return

            for sub_name, sub_module in module._modules.items():
                new_name = sub_name if name is None else "{}.{}".format(name, sub_name)
                    
                iterate_module_to_register_hook(sub_module, new_name)

        iterate_module_to_register_hook(self)

        if not self.hook_registered:
            raise Exception("Layer Path {} Not Found".format(self.gradcam_layer))

    def forward(self, input1, input2, output_shape=(224, 224)):
        """Calculates GradCAM activation maps for a pair of images.

        Args:
            input1 (torch.Tensor): First image tensor.
            input2 (torch.Tensor): Second image tensor.
            output_shape (tuple, optional): Resolution of output heatmaps. Defaults to (224, 224).

        Returns:
            tuple: (att, v_scores, emb_vecs, activation_maps)
                activation_maps (list[torch.Tensor]): Heatmaps for each attention group.
        """
        self.eval()
        inputs = torch.cat((input1.unsqueeze(0), input2.unsqueeze(0)), 0)

        _, _, att, v_scores, emb_vecs, emb_vec_gates = self._forward(inputs)

        activation_maps = []
        for i in range(self.attention_num):
            outs = emb_vecs[:, i]
            gates = emb_vec_gates[:, i]

            emb_vec_weights = get_emb_similarity_vector(outs[0], gates[0],
                                                        outs[1], gates[1])

            results = torch.FloatTensor()

            for out_index in range(len(outs)):
                out = outs[out_index]

                grad = None

                for emb_vec_idx in range(len(emb_vec_weights)):
                    emb_weight = emb_vec_weights[emb_vec_idx] * out[emb_vec_idx]
                    self.zero_grad()
                    out[emb_vec_idx].backward(retain_graph=True)
                    emb_grad = self.backward_result[out_index] * emb_weight
                    if grad is None:
                        grad = emb_grad
                    else:
                        grad += emb_grad

                result = torch.sum(grad * self.forward_result[out_index], dim=0).cpu()
                result = F.interpolate(result.unsqueeze(0).unsqueeze(0), output_shape, mode="bilinear")
                result = torch.relu(result) / torch.max(result)
                results = torch.cat((results, result.cpu().detach().squeeze().unsqueeze(0)), 0)

            activation_maps.append(results)

        return att, v_scores, emb_vecs, emb_vec_gates, activation_maps

    def forward_hook(self, _, input_, output):
        """Hook to capture forward activations.

        Args:
            input_ (torch.Tensor): Input to the layer.
            output (torch.Tensor): Output from the layer.
        """
        self.forward_result = output

    def backward_hook(self, _, grad_input, grad_output):
        """Hook to capture backward gradients.

        Args:
            grad_input (torch.Tensor): Gradients with respect to input.
            grad_output (torch.Tensor): Gradients with respect to output.
        """
        self.backward_result = grad_output[0]

