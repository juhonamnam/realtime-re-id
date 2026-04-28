import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from pycocotools.coco import COCO
from pycocotools.mask import decode
from src.utils.file_path import get_dataset_path, get_tmp_path
from src.utils.segment_info import get_segment_groups
import os
from .transforms.LetterboxPad import LetterboxPad
from .transforms.RandomCrop import get_random_crop_params
from .transforms.RandomHorizontalFlip import should_horizontal_flip
from .transforms.RandomResolutionReduce import RandomResolutionReduce

DATASET_NAME = "coco"


def get_data_path():
    return {
        "train": {
            "annotation": get_dataset_path(DATASET_NAME, "densepose_coco_2014_train.json"),
            "image_dir": get_dataset_path(DATASET_NAME, "train2014"),
            "cache_dir": get_tmp_path(DATASET_NAME, "train2014"),
        },
        "val": {
            "annotation": get_dataset_path(DATASET_NAME, "densepose_coco_2014_valminusminival.json"),
            "image_dir": get_dataset_path(DATASET_NAME, "val2014"),
            "cache_dir": get_tmp_path(DATASET_NAME, "val2014"),
        },
        "test": {
            "annotation": get_dataset_path(DATASET_NAME, "densepose_coco_2014_minival.json"),
            "image_dir": get_dataset_path(DATASET_NAME, "val2014"),
            "cache_dir": get_tmp_path(DATASET_NAME, "val2014"),
        },
    }

class SegTransform(nn.Module):
    def __init__(self, image_resolution, generator=None,
                 random_crop=True, random_horizontal_flip=True,
                 random_resolution_reduce=True):
        super().__init__()
        self.generator = generator
        self.random_crop = random_crop
        self.random_horizontal_flip = random_horizontal_flip

        t = []

        t.append(LetterboxPad(target_size=image_resolution))

        if random_resolution_reduce:
            t.append(RandomResolutionReduce(target_size=image_resolution, generator=generator))

        self.image_transform = transforms.Compose(t)
        self.mask_transform = LetterboxPad(target_size=image_resolution,
                                           interpolation=transforms.InterpolationMode.NEAREST)

    def forward(self, image, masks):
        if self.random_crop:
            do_crop, params = get_random_crop_params(image.size()[1:], generator=self.generator)
            if do_crop:
                image = TF.crop(image, *params)
                masks = map(lambda m: TF.crop(m, *params), masks)

        image = self.image_transform(image)

        masks = list(map(lambda m: self.mask_transform(m), masks))

        if self.random_horizontal_flip:
            do_flip = should_horizontal_flip(generator=self.generator)
            if do_flip:
                image = TF.hflip(image)
                masks = list(map(lambda m: TF.hflip(m), masks))

        return image, masks

def get_train_dataset(image_resolution,
                      seg_variant,
                      generator=None,
                      random_crop=True,
                      random_horizontal_flip=True,
                      random_resolution_reduce=True):
    annotation_path = get_data_path()["train"]["annotation"]
    image_dir = get_data_path()["train"]["image_dir"]
    cache_dir = get_data_path()["train"]["cache_dir"]
    coco = COCO(annotation_path)

    transform = SegTransform(image_resolution=image_resolution,
                             generator=generator,
                             random_crop=random_crop,
                             random_horizontal_flip=random_horizontal_flip,
                             random_resolution_reduce=random_resolution_reduce)
    return COCODataset(seg_variant, transform, coco, image_dir,
                       cache_dir, image_resolution)

def get_val_dataset(image_resolution,
                    seg_variant,
                    generator=None,
                    random_crop=True,
                    random_horizontal_flip=True,
                    random_resolution_reduce=True):
    annotation_path = get_data_path()["val"]["annotation"]
    image_dir = get_data_path()["val"]["image_dir"]
    cache_dir = get_data_path()["val"]["cache_dir"]
    coco = COCO(annotation_path)

    transform = SegTransform(image_resolution=image_resolution,
                             generator=generator,
                             random_crop=random_crop,
                             random_horizontal_flip=random_horizontal_flip,
                             random_resolution_reduce=random_resolution_reduce)
    return COCODataset(seg_variant, transform, coco, image_dir,
                       cache_dir, image_resolution)

def get_test_dataset(image_resolution,
                     seg_variant,
                     generator=None,
                     random_crop=True,
                     random_horizontal_flip=True,
                     random_resolution_reduce=True):
    annotation_path = get_data_path()["test"]["annotation"]
    image_dir = get_data_path()["test"]["image_dir"]
    cache_dir = get_data_path()["test"]["cache_dir"]
    coco = COCO(annotation_path)

    transform = SegTransform(image_resolution=image_resolution,
                             generator=generator,
                             random_crop=random_crop,
                             random_horizontal_flip=random_horizontal_flip,
                             random_resolution_reduce=random_resolution_reduce)
    return COCODataset(seg_variant, transform, coco, image_dir,
                       cache_dir, image_resolution)


class COCODataset(dataset.Dataset):
    def __init__(self, seg_variant, transform, coco, image_dir,
                 cache_dir, image_resolution):
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.loader = default_loader
        self.coco = coco
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.image_resolution = image_resolution

        self.segment_groups = get_segment_groups(seg_variant)

        image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds(catNms=['person']))

        ann_ids = coco.getAnnIds(imgIds=image_ids, iscrowd=False, catIds=self.coco.getCatIds(catNms=['person']))

        self.anns = [ann for ann in self.coco.loadAnns(ann_ids) if self.filter_ann(ann)]

    def _getitem(self, ann):
        [x_start, y_start, width, height] = ann["bbox"]

        img_cache_file = f"{self.cache_dir}/{ann['id']}_img.pt"

        if os.path.isfile(img_cache_file):
            img = torch.load(img_cache_file, weights_only=True)
        else:
            image_id = ann["image_id"]
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = f"{self.image_dir}/{image_info['file_name']}"
            image = self.loader(image_path)
            img = image.crop((x_start, y_start, x_start + width, y_start + height))
            img = self.to_tensor(img)

            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(img, img_cache_file)

        dp_masks = ann["dp_masks"]

        masks = []

        for i in range(len(dp_masks)):
            dp_mask = dp_masks[i]
            if len(dp_mask):
                mask = torch.tensor(decode(dp_mask), dtype=torch.float)
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                     size=(int(height), int(width)),
                                     mode="nearest").squeeze(0)
            else:
                mask = torch.zeros(self.image_resolution, dtype=torch.float).unsqueeze(0)
            masks.append(mask)

        img, masks = self.transform(img, masks)

        masks = torch.cat(masks, dim=0)

        inner_seg = []
        seg = []

        for segment_group in self.segment_groups:
            if segment_group["is_background"]:
                seg.append("background")
            else:
                segment = masks[segment_group["dp_mask_indices"]].sum(dim=0)
                segment = segment.unsqueeze(0)
                seg.append(segment)
                inner_seg.append(segment)

        inner_seg = torch.cat(inner_seg, dim=0).sum(dim=0, keepdim=True)
        bg = 1 - inner_seg

        for i, s in enumerate(seg):
            if s == "background":
                seg[i] = bg

        seg = torch.cat(seg, dim=0)

        return img, seg

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]
        ann = self.anns[index]
        if isinstance(ann, list):
            return [self._getitem(a) for a in ann]
        return self._getitem(ann)


    def __len__(self):
        return len(self.anns)

    @staticmethod
    def filter_ann(ann):
        [_, _, width, height] = ann["bbox"]
        if width < 25 or height < 75:
            return False

        if "dp_masks" not in ann:
            return False

        dp_masks_count = 0

        for dm in ann["dp_masks"]:
            if len(dm) >= 1:
                dp_masks_count += 1

        if dp_masks_count < 1:
            return False

        return True
