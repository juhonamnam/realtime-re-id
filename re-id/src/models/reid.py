import torch
import torch.nn as nn
import torch.nn.functional as F
from src.metric import get_part_distances

from src.utils.segment_info import get_segment_groups, get_attention_groups
from .mobilenetv3 import MobilenetV3
from .hrnet import hrnet32
from .resnet import ResNet
from src.utils.file_path import get_weight_file_path, get_export_file_path, PRETRAINED_PATH
from src.utils.split_state_dict import load_state_dict, save_state_dict

__all__ = ['ReIDModel']


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

        self.use_foreground = any([x["is_foreground"]
                                  for x in self.attention_groups])

        self.seg_to_part_indices = [x["seg_idx"]
                                    for x in self.attention_groups if not x["is_foreground"]]

        self.model_name = self.get_model_name(
            model_variant, seg_variant, emb_len, input_resolution)

        self.onnx_export_file_path = get_export_file_path(
            self.model_name, "model.onnx")

        if model_variant == "m3small":
            self.backbone = MobilenetV3(variant="small", pretrained=backbone_pretrained,
                                        use_fpn=True, fpn_out_ch=256)
        elif model_variant == "m3large":
            self.backbone = MobilenetV3(variant="large", pretrained=backbone_pretrained,
                                        use_fpn=True, fpn_out_ch=256)
        elif model_variant == "hrnet32":
            self.backbone = hrnet32(pretrained=backbone_pretrained)

        elif model_variant == "resnet50":
            self.backbone = ResNet(variant="resnet50", pretrained=backbone_pretrained,
                                   use_fpn=True, fpn_out_ch=256)

        else:
            raise Exception(f"\"{model_variant}\" Not Supported")

        self.segmentation = nn.Sequential(
            Segmentation(ch_out=self.backbone.out_ch,
                         segment_num=self.segment_num),
            nn.Softmax2d(),
        )

        self.embeddings = Embedding(
            input_dim=self.backbone.out_ch, output_dim=emb_len)

        self.gradcam_layer = "embeddings"

        if pretrained:
            try:
                state_dict = load_state_dict(PRETRAINED_PATH, self.model_name)
                self.load_state_dict(state_dict)
            except Exception as e:
                print(f"Failed to load pretrained weight: {e}")

    @staticmethod
    def get_model_name(model_variant, seg_variant, emb_len, input_resolution):
        model_name = "_".join(["reid",
                               model_variant,
                               f"{seg_variant}{emb_len}e",
                               f"{input_resolution[1]}x{input_resolution[0]}"])
        return model_name

    def clear_seg_noise(self, seg: torch.Tensor):
        # batch, 1, height, width
        seg_max = seg.max(dim=1, keepdim=True)

        # batch, segmentation_num, height, width
        seg = torch.zeros_like(seg)
        # batch, segmentation_num, height, width
        seg = seg.scatter(1, seg_max.indices, seg_max.values)
        return seg

    def forward_embedding(self, embedding_feat: torch.Tensor, seg: torch.Tensor):
        # batch, part_num, height, width
        att = seg[:, self.seg_to_part_indices]

        clean_seg = self.clear_seg_noise(seg)
        # batch, part_num
        v_scores = clean_seg[:, self.seg_to_part_indices].amax(dim=(2, 3))

        # batch, 1, emb_len, height, width
        embedding_feat = embedding_feat.unsqueeze(1)

        # batch, part_num, emb_len, height, width
        emb_vecs = att.unsqueeze(2) * embedding_feat
        # batch, part_num, emb_len
        emb_vecs = emb_vecs.sum(dim=(3, 4)) / \
            att.sum(dim=(2, 3)).unsqueeze(2).clamp(min=0.1)

        if self.use_foreground:
            # batch, 1, height, width
            f_att = att.sum(dim=1, keepdim=True)

            clean_f_att = (f_att > 0.5) * f_att
            # batch, 1
            f_v_score = clean_f_att.amax(dim=(2, 3))

            # batch, 1, emb_len, height, width
            f_emb_vecs = f_att.unsqueeze(2) * embedding_feat
            # batch, 1, emb_len
            f_emb_vecs = f_emb_vecs.sum(
                dim=(3, 4)) / f_att.sum(dim=(2, 3)).unsqueeze(2).clamp(min=0.1)

            # batch, part_num + 1, height, width
            att = torch.cat((att, f_att), dim=1)
            # batch, part_num + 1
            v_scores = torch.cat((v_scores, f_v_score), dim=1)
            # batch, part_num + 1, emb_len
            emb_vecs = torch.cat((emb_vecs, f_emb_vecs), dim=1)

        return att, v_scores, emb_vecs

    def forward(self, x: torch.Tensor):
        spatial_feat = self.backbone(x)

        seg = self.segmentation(spatial_feat)
        embedding_feat = self.embeddings(spatial_feat)
        att, v_scores, emb_vecs = self.forward_embedding(embedding_feat, seg)

        return (seg, att), (v_scores, emb_vecs)

    def save_pretrained(self):
        save_state_dict(self.state_dict(), PRETRAINED_PATH, self.model_name)

    def export(self):
        export_model = ReIDExportModel(self)
        return export_model

    def get_train_model(self, class_nums: list[int]):
        return ReIDTrainModel(self, class_nums)

    def get_gradcam_model(self):
        return ReIDGradCAM(self)


class ReIDTrainModel(nn.Module):
    def __init__(self, reid_model: ReIDModel, class_nums: list[int]):
        super().__init__()
        self.reid_model = reid_model

        self.concat_logits = nn.ModuleList([BNClassifier(
            reid_model.emb_len * reid_model.attention_num, num)
            for num in class_nums])

        self.foreground_logits = nn.ModuleList([BNClassifier(
            reid_model.emb_len, num) for num in class_nums])

        self.gap_logits = nn.ModuleList([BNClassifier(
            reid_model.emb_len, num) for num in class_nums])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        cls_num_str = "x".join([str(num) for num in class_nums])
        self.weight_file_path = get_weight_file_path(
            f"{self.reid_model.model_name}_{cls_num_str}cls_epoch_{{}}.pt")

    def forward_segmentation(self, x):
        spatial_feat = self.reid_model.backbone(x)
        return self.reid_model.segmentation(spatial_feat)

    def forward(self, x, seg):
        spatial_feat = self.reid_model.backbone(x)

        embedding_feat = self.reid_model.embeddings(spatial_feat)
        att, _, emb_vecs = self.reid_model.forward_embedding(
            embedding_feat, seg)

        concat_emb_vecs = emb_vecs.flatten(1)
        concat_logits = [classifier(concat_emb_vecs)
                         for classifier in self.concat_logits]

        if self.reid_model.use_foreground:
            f_feat = emb_vecs[:, -1]
        else:
            f_att = att.sum(dim=1, keepdim=True)
            f_feat = embedding_feat * f_att
            f_feat = f_feat.sum(dim=(2, 3)) / \
                f_att.sum(dim=(2, 3)).clamp(min=0.1)

        foreground_logits = [classifier(f_feat)
                             for classifier in self.foreground_logits]

        gap_feat = self.avgpool(embedding_feat).flatten(1)
        gap_logits = [classifier(gap_feat) for classifier in self.gap_logits]

        return emb_vecs, concat_logits, foreground_logits, gap_logits


class ReIDExportModel(ReIDModel):
    def __init__(self, reid_model: ReIDModel):
        super().__init__(reid_model.model_variant,
                         reid_model.seg_variant,
                         reid_model.emb_len,
                         reid_model.input_resolution)
        self.load_state_dict(reid_model.state_dict())
        self.eval()

    def forward(self, x):
        _, (v_scores, emb_vecs) = super().forward(x)
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
            if module is None:
                return
            if name == self.gradcam_layer:
                module.register_forward_hook(self.forward_hook)
                module.register_backward_hook(self.backward_hook)
                self.hook_registered = True
                return

            for sub_name, sub_module in module._modules.items():
                new_name = sub_name if name is None else "{}.{}".format(
                    name, sub_name)

                iterate_module_to_register_hook(sub_module, new_name)

        iterate_module_to_register_hook(self)

        if not self.hook_registered:
            raise Exception(
                "Layer Path {} Not Found".format(self.gradcam_layer))

    def forward(self, input1, input2, output_shape=(224, 224)):
        self.eval()
        inputs = torch.cat((input1.unsqueeze(0), input2.unsqueeze(0)), 0)

        (_, att), (v_scores, emb_vecs) = super().forward(inputs)

        part_distances = get_part_distances(emb_vecs[0], emb_vecs[1])

        activation_maps = []
        for i in range(self.attention_num):
            self.zero_grad()
            part_distances[i].backward(retain_graph=True)
            grads = self.backward_result
            grads = grads.sum(dim=1)
            grads = F.interpolate(grads.unsqueeze(
                0), output_shape, mode="bilinear").squeeze(0)
            grads = torch.relu(grads) / grads.amax(dim=(1, 2),
                                                   keepdim=True).clamp(min=1e-6)

            activation_maps.append(grads)

        return att, v_scores, emb_vecs, activation_maps

    def forward_hook(self, _, input_, output):
        self.forward_result = output

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = grad_output[0]


class BNClassifier(nn.Module):
    # Source: https://github.com/upgirlnana/Pytorch-Person-REID-Baseline-Bag-of-Tricks
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        # BoF: this doesn't have a big impact on perf according to author on github
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self._init_params()

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return cls_score

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # ResNet = 0.01, Bof and ISP-reid = 0.001
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Segmentation(nn.Module):
    def __init__(self, ch_out, segment_num):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(ch_out)
        self.classifier = nn.Conv2d(
            in_channels=ch_out, out_channels=segment_num, kernel_size=1, stride=1, padding=0)
        self._init_params()

    def forward(self, x):
        x = self.bn(x)
        return self.classifier(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # ResNet = 0.01, Bof and ISP-reid = 0.001
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                input_dim, output_dim, 1, stride=1, padding=0
            )
        )
        layers.append(nn.BatchNorm2d(output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        return self.layers(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
