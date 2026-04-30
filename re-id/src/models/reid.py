import torch
import torch.nn as nn
import torch.nn.functional as F
from src.metric import get_emb_similarity_vector

from torchvision.ops.misc import Conv2dNormActivation

from src.utils.segment_info import get_segment_groups, get_attention_groups
from .mobilenetv3 import MobilenetV3
from src.utils.file_path import get_weight_file_path, get_pretrained_file_path, get_export_file_path


# ImageNet Normalization
class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class ReIDModel(nn.Module):
    def __init__(self,
                 model_variant,
                 seg_variant,
                 emb_len,
                 input_resolution,
                 pretrained=False,
                 backbone_pretrained=False):
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

        self.model_name = "_".join(["reid",
                                    model_variant,
                                    f"{self.attention_num}a{self.emb_len}e",
                                    f"{input_resolution[1]}x{input_resolution[0]}"])

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


    def _forward(self, x: torch.Tensor):
        x = self.normalize(x)
        fused_feat, local_feat = self.backbone(x)                                # batch, out_ch, height, width
        seg = self.segmentation(fused_feat)                                      # batch, segmentation_num, height, width

        raw_att = seg[:, self.seg_to_att_indices]                                # batch, attention_num, height, width

        att_max = raw_att.max(dim=1, keepdim=True)                               # batch, 1, height, width 

        att = torch.zeros_like(raw_att)                                          # batch, attention_num, height, width
        att = att.scatter(1, att_max.indices, att_max.values)                    # batch, attention_num, height, width

        v_scores = att.amax(dim=(2, 3))                                          # batch, attention_num

        emb_vecs = []
        for i in range(self.attention_num):
            emb_vec = self.emb_vec[i](local_feat)                                # batch, emb_len, height, width

            att_weight = att[:, [i]]                                             # batch, 1, height, width

            emb_vec = att_weight * emb_vec                                       # batch, emb_len, height, width
            emb_vec = emb_vec.sum(dim=(2, 3))                                    # batch, emb_len
            emb_vecs.append(emb_vec)
        emb_vecs = torch.stack(emb_vecs, dim=1)                                  # batch, attention_num, feature_len

        return local_feat, seg, att, v_scores, emb_vecs

    def forward(self, x: torch.Tensor):
        _, seg, att, v_scores, emb_vecs = self._forward(x)
        return (seg, att), (v_scores, emb_vecs)

    def export(self):
        export_model = ReIDExportModel(self)
        return export_model

    def get_train_model(self, class_nums: list[int]):
        return ReIDTrainModel(self, class_nums)

    def get_gradcam_model(self):
        return ReIDGradCAM(self)


class ReIDTrainModel(nn.Module):
    def __init__(self, reid_model: ReIDModel, class_num: int):
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
        for p in self.reid_model.backbone.parameters():
            p.requires_grad = True

        for p in self.reid_model.segmentation.parameters():
            p.requires_grad = True

    def train_segmentation(self):
        for p in self.reid_model.backbone.parameters():
            p.requires_grad = False

        for p in self.reid_model.segmentation.parameters():
            p.requires_grad = True

    def train_re_id(self):
        for p in self.reid_model.backbone.parameters():
            p.requires_grad = False

        for p in self.reid_model.segmentation.parameters():
            p.requires_grad = False

    def forward(self, x):
        local_feat, seg, att, v_scores, emb_vecs = self.reid_model._forward(x)

        global_feat = self.global_features(local_feat)
        global_att = att.sum(dim=1, keepdim=True)

        global_feat = global_feat * global_att
        global_feat = global_feat.sum(dim=(2, 3))

        ft = emb_vecs / att.sum(dim=(2, 3)).unsqueeze(-1).clamp(min=1e-12)
        ft = ft * v_scores.unsqueeze(-1)

        global_feat = global_feat / global_att.sum(dim=(2, 3)).clamp(min=1e-12)

        global_v_score = global_att.amax(dim=(2, 3))
        global_feat = global_feat * global_v_score

        concat_feature = torch.cat([global_feat,
                                    ft.flatten(1)],
                                   dim=1)

        class_logits = self.logit(concat_feature)

        return seg, v_scores, emb_vecs, class_logits


class ReIDExportModel(ReIDModel):
    def __init__(self, reid_model: ReIDModel):
        super().__init__(reid_model.model_variant,
                         reid_model.seg_variant,
                         reid_model.emb_len,
                         reid_model.input_resolution)
        self.load_state_dict(reid_model.state_dict())
        self.eval()

    def forward(self, x):
        _, _, _, v_scores, emb_vecs = super()._forward(x)
        return v_scores, emb_vecs


class ReIDGradCAM(ReIDModel):
    def __init__(self, reid_model: ReIDModel):
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
        self.eval()
        inputs = torch.cat((input1.unsqueeze(0), input2.unsqueeze(0)), 0)

        _, _, att, v_scores, emb_vecs = self._forward(inputs)

        activation_maps = []
        for i in range(self.attention_num):
            outs = emb_vecs[:, i]

            emb_vec_weights = get_emb_similarity_vector(outs[0], outs[1])

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

        return att, v_scores, emb_vecs, activation_maps

    def forward_hook(self, _, input_, output):
        self.forward_result = output

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = grad_output[0]

