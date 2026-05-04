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

__all__ = ['get_dataset']

DATASET_TYPE = "re_id"

def get_data_path():
    """Returns the image and annotation directory paths for different stages.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' paths.
    """
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
    """Parses an XML annotation file for Re-ID.

    Args:
        annotation_path (str): Path to the XML file.

    Returns:
        tuple: (image_path, id, camera)
            image_path (str): Relative path to the image.
            id (str): Person ID.
            camera (str): Camera ID.
    """
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
    """Data augmentation and preprocessing for Person Re-ID.

    Attributes:
        transforms (transforms.Compose): Combined sequence of transforms.
    """
    def __init__(self, image_resolution, generator, random_crop,
                 random_horizontal_flip, random_erasing,
                 random_resolution_reduce):
        """Initializes ReIDTransform.

        Args:
            image_resolution (tuple): Target (H, W).
            generator (torch.Generator): Random generator.
            random_crop (bool): Whether to apply random crop.
            random_horizontal_flip (bool): Whether to apply random flip.
            random_erasing (bool): Whether to apply random erasing.
            random_resolution_reduce (bool): Whether to apply resolution reduction.
        """
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
        """Applies transforms to an image.

        Args:
            image (PIL.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        return self.transforms(image)

def get_dataset(image_resolution,
                data_stage,
                generator=None,
                random_crop=True,
                random_horizontal_flip=True,
                random_erasing=False,
                random_resolution_reduce=True):
    """Helper function to create a REIDDataset instance.

    Args:
        image_resolution (tuple): Target resolution.
        data_stage (str): Split name ('train', 'val', or 'test').
        generator (torch.Generator, optional): Random generator.
        random_crop (bool, optional): Enable cropping.
        random_horizontal_flip (bool, optional): Enable flipping.
        random_erasing (bool, optional): Enable erasing.
        random_resolution_reduce (bool, optional): Enable resolution reduction.

    Returns:
        REIDDataset: The created dataset instance.
    """
    data_path = get_data_path()[data_stage]

    transform = ReIDTransform(image_resolution=image_resolution,
                              generator=generator,
                              random_crop=random_crop,
                              random_horizontal_flip=random_horizontal_flip,
                              random_erasing=random_erasing,
                              random_resolution_reduce=random_resolution_reduce)

    return REIDDataset(transform, data_path)


class REIDDataset(dataset.Dataset):
    """Dataset class for Person Re-Identification.

    Attributes:
        transform (ReIDTransform): Transform module.
        loader (Callable): Image loader function.
        data_path (dict): Dictionary with image and annotation paths.
        image_anns (list[dict]): List of image annotations.
        _id2label (dict): Mapping from person ID to label index.
        _id2index (dict): Mapping from person ID to list of dataset indices.
    """
    def __init__(self, transform, data_path):
        """Initializes REIDDataset.

        Args:
            transform (ReIDTransform): Transform module.
            data_path (dict): Dictionary with 'image_dir' and 'annotation' paths.
        """

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
        """Returns all dataset indices associated with a specific person ID.

        Args:
            id (str): Person ID.

        Returns:
            list[int]: List of indices.
        """
        return self._id2index[id]

    def get_camera_by_index(self, index):
        """Returns the camera ID for a given dataset index.

        Args:
            index (int): Dataset index.

        Returns:
            str: Camera ID.
        """
        return self.image_anns[index]["camera"]

    def _getitem(self, img_ann):
        """Loads and transforms an image for a given annotation.

        Args:
            img_ann (dict): Image annotation dictionary.

        Returns:
            tuple[torch.Tensor, int]: Transformed image and label index.
        """
        path = os.path.join(self.data_path["image_dir"], img_ann["image_path"])
        id_label = self._id2label[img_ann["id"]]

        image = self.loader(path)
        image = self.transform(image)

        return image, id_label

    def __getitem__(self, index):
        """Returns the item at the given index.

        Args:
            index (int or list): Dataset index.

        Returns:
            tuple or list: Data items.
        """
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]
        img_ann = self.image_anns[index]
        if isinstance(img_ann, list):
            return [self._getitem(i) for i in img_ann]
        return self._getitem(img_ann)

    def __len__(self):
        """Returns the total number of items in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.image_anns)

    @property
    def ids(self):
        """Returns a list of person IDs corresponding to dataset images.

        Returns:
            list[str]: Person IDs.
        """
        return [img["id"] for img in self.image_anns]

    @property
    def unique_ids(self):
        """Returns a sorted list of unique person IDs in the dataset.

        Returns:
            list[str]: Unique person IDs.
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """Returns a list of camera IDs corresponding to dataset images.

        Returns:
            list[str]: Camera IDs.
        """
        return [img_ann["camera"] for img_ann in self.image_anns]

    @staticmethod
    def list_image_annotations(annotation_dir):
        """Lists and parses all XML annotations in a directory.

        Args:
            annotation_dir (str): Directory containing XML files.

        Returns:
            list[dict]: List of parsed annotations.
        """
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