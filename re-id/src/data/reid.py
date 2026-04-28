import torch.nn as nn
from torchvision import transforms
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from .transforms.RandomErasing import RandomErasing
from .transforms.RandomHorizontalFlip import RandomHorizontalFlip
from .transforms.RandomCrop import RandomCrop
from .transforms.LetterboxPad import LetterboxPad
from .transforms.RandomResolutionReduce import RandomResolutionReduce
import os
import re
import collections
from src.utils.file_path import get_dataset_path
import xml.etree.ElementTree as ET

DATASET_TYPE = "re_id"

def get_data_path():
    return {
        "train": {
            "image_dir": get_dataset_path(DATASET_TYPE, "train", "images"),
            "annotation": get_dataset_path(DATASET_TYPE, "train", "labels"),
        },
        "val": {
            "image_dir": get_dataset_path(DATASET_TYPE, "val", "images"),
            "annotation": get_dataset_path(DATASET_TYPE, "val", "labels"),
        },
        "test": {
            "image_dir": get_dataset_path(DATASET_TYPE, "val", "images"),
            "annotation": get_dataset_path(DATASET_TYPE, "val", "labels"),
        },
    }

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    image_path = None
    id = None
    camera = None

    try:
        for child in root:
            if child.tag == "FILE":
                image_path = child.find("name").text
            elif child.tag == "OBJECT":
                id = child.attrib["ID"]
            elif child.tag == "CAMERA":
                camera = child.attrib["ID"]
    except Exception as e:
        print(f"Error parsing {annotation_path}: {e}")
    
    return image_path, id, camera

class ReIDTransform(nn.Module):
    def __init__(self, image_resolution, generator, random_crop,
                 random_horizontal_flip, random_erasing,
                 random_resolution_reduce):
        super().__init__()

        t = []

        if random_crop:
            t.append(RandomCrop(generator=generator))

        t.append(transforms.ToTensor())

        if random_erasing:
            t.append(RandomErasing(generator=generator))

        t.append(LetterboxPad(target_size=image_resolution))

        if random_resolution_reduce:
            t.append(RandomResolutionReduce(target_size=image_resolution, generator=generator))

        if random_horizontal_flip:
            t.append(RandomHorizontalFlip(generator=generator))

        self.transforms = transforms.Compose(t)

    def forward(self, image):
        return self.transforms(image)

def get_dataset(image_resolution,
                data_stage,
                generator=None,
                random_crop=True,
                random_horizontal_flip=True,
                random_erasing=False,
                random_resolution_reduce=True):
    data_path = get_data_path()[data_stage]

    transform = ReIDTransform(image_resolution=image_resolution,
                              generator=generator,
                              random_crop=random_crop,
                              random_horizontal_flip=random_horizontal_flip,
                              random_erasing=random_erasing,
                              random_resolution_reduce=random_resolution_reduce)

    return REIDDataset(transform, data_path)


class REIDDataset(dataset.Dataset):
    def __init__(self, transform, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        self.image_anns = self.list_image_annotations(data_path["annotation"])

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

        id2index = collections.defaultdict(list)
        for idx, id in enumerate(self.ids):
            id2index[id].append(idx)
        self._id2index = id2index

    def get_indexes_by_id(self, id):
        return self._id2index[id]

    def get_camera_by_index(self, index):
        return self.image_anns[index]["camera"]

    def _getitem(self, img_ann):
        path = os.path.join(self.data_path["image_dir"], img_ann["image_path"])
        id_label = self._id2label[img_ann["id"]]

        image = self.loader(path)
        image = self.transform(image)

        return image, id_label

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]
        img_ann = self.image_anns[index]
        if isinstance(img_ann, list):
            return [self._getitem(i) for i in img_ann]
        return self._getitem(img_ann)

    def __len__(self):
        return len(self.image_anns)

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [img["id"] for img in self.image_anns]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [img_ann["camera"] for img_ann in self.image_anns]

    @staticmethod
    def list_image_annotations(annotation_dir):
        annotation_paths = [os.path.join(root, f)
                            for root, _, files in os.walk(annotation_dir) for f in files
                            if re.match(r'^.*\.(xml)$', f)]
        annotation_paths.sort()

        image_labels = []
        for annotation_path in annotation_paths:
            image_path, id, camera = parse_annotation(annotation_path)

            if image_path is None:
                print(f"image_path is not found for {annotation_path}")
                continue
            if id is None:
                print(f"id is not found for {annotation_path}")
                continue
            if camera is None:
                print(f"camera is not found for {annotation_path}")
                continue

            image_labels.append({"image_path": image_path,
                                 "id": id,
                                 "camera": camera})

        return image_labels