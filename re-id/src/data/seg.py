from torchvision import transforms
import torch.nn.functional as F
import torch
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from pycocotools.coco import COCO
from pycocotools.mask import decode
from src.utils.file_path import get_dataset_path, get_tmp_path
from src.utils.segment_info import get_segment_groups
import os
from .transforms.LetterboxPad import LetterboxPad

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

def get_image_transform(image_resolution):
    return transforms.Compose([
        transforms.ToTensor(),
        LetterboxPad(target_size=image_resolution),
    ])

def get_mask_transform(image_resolution):
    return LetterboxPad(target_size=image_resolution,
                        interpolation=transforms.InterpolationMode.NEAREST)

def get_train_dataset(image_resolution, seg_variant):
    annotation_path = get_data_path()["train"]["annotation"]
    image_dir = get_data_path()["train"]["image_dir"]
    cache_dir = get_data_path()["train"]["cache_dir"]
    coco = COCO(annotation_path)

    image_transform = get_image_transform(image_resolution)
    mask_transform = get_mask_transform(image_resolution)
    return COCODataset(seg_variant, image_transform, mask_transform, coco, image_dir, cache_dir, image_resolution)

def get_val_dataset(image_resolution, seg_variant):
    annotation_path = get_data_path()["val"]["annotation"]
    image_dir = get_data_path()["val"]["image_dir"]
    cache_dir = get_data_path()["val"]["cache_dir"]
    coco = COCO(annotation_path)

    image_transform = get_image_transform(image_resolution)
    mask_transform = get_mask_transform(image_resolution)
    return COCODataset(seg_variant, image_transform, mask_transform, coco, image_dir, cache_dir, image_resolution)

def get_test_dataset(image_resolution, seg_variant):
    annotation_path = get_data_path()["test"]["annotation"]
    image_dir = get_data_path()["test"]["image_dir"]
    cache_dir = get_data_path()["test"]["cache_dir"]
    coco = COCO(annotation_path)

    image_transform = get_image_transform(image_resolution)
    mask_transform = get_mask_transform(image_resolution)
    return COCODataset(seg_variant, image_transform, mask_transform, coco, image_dir, cache_dir, image_resolution)


class COCODataset(dataset.Dataset):
    def __init__(self, seg_variant, image_transform, mask_transform, coco, image_dir, cache_dir, image_resolution):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
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

        img_cache_file = f"{self.cache_dir}/{ann['id']}_img_{self.image_resolution[1]}x{self.image_resolution[0]}.pt"

        if os.path.isfile(img_cache_file):
            img = torch.load(img_cache_file, weights_only=True)
        else:
            image_id = ann["image_id"]
            image_info = self.coco.loadImgs(image_id)[0]
            image_path = f"{self.image_dir}/{image_info['file_name']}"
            image = self.loader(image_path)
            cropped_image = image.crop((x_start, y_start, x_start + width, y_start + height))
            img = self.image_transform(cropped_image)

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
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros(self.image_resolution, dtype=torch.float).unsqueeze(0)
            masks.append(mask)

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
