from torchvision.datasets.folder import default_loader
import os
import re
import collections
from src.utils.file_path import get_dataset_path
import xml.etree.ElementTree as ET

from .abstract import ReIDDataset

__all__ = ['NIADataset']

DATASET_NAME = "nia"


def get_data_path():
    """Returns the image and annotation directory paths for different stages.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' paths.
    """
    return {
        "train": {
            "image_dir": get_dataset_path(DATASET_NAME, "train", "images"),
            "annotation": get_dataset_path(DATASET_NAME, "train", "labels"),
        },
        "val": {
            "image_dir": get_dataset_path(DATASET_NAME, "val", "images"),
            "annotation": get_dataset_path(DATASET_NAME, "val", "labels"),
        },
        "test": {
            "image_dir": get_dataset_path(DATASET_NAME, "val", "images"),
            "annotation": get_dataset_path(DATASET_NAME, "val", "labels"),
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


class NIADataset(ReIDDataset):
    """Dataset class for Person Re-Identification.

    Attributes:
        transform (ReIDTransform): Transform module.
        loader (Callable): Image loader function.
        data_path (dict): Dictionary with image and annotation paths.
        image_anns (list[dict]): List of image annotations.
        _id2label (dict): Mapping from person ID to label index.
        _id2index (dict): Mapping from person ID to list of dataset indices.
    """
    name = DATASET_NAME

    def __init__(self, transform, stage, *args, **kwargs):
        """Initializes REIDDataset.

        Args:
            transform (ReIDTransform): Transform module.
            data_path (dict): Dictionary with 'image_dir' and 'annotation' paths.
        """
        super().__init__(transform, stage, *args, **kwargs)

        self.loader = default_loader
        self.data_path = get_data_path()[stage]

        self.image_anns = self.list_image_annotations(
            self.data_path["annotation"])

        self.ids = [img["id"] for img in self.image_anns]
        self._unique_ids = sorted(set(self.ids))
        self.cameras = [img_ann["camera"] for img_ann in self.image_anns]

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

    def _getitem(self, index):
        """Loads an image for a given annotation.

        Args:
            index (int): Dataset index.

        Returns:
            tuple: (image, id_label)
                image (PIL.Image): Loaded image.
                id_label (int): Label index corresponding to the person ID.
        """
        img_ann = self.image_anns[index]

        path = os.path.join(self.data_path["image_dir"], img_ann["image_path"])
        id_label = self._id2label[img_ann["id"]]

        image = self.loader(path)

        return image, id_label

    def __len__(self):
        """Returns the total number of items in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.image_anns)

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

    @property
    def unique_ids(self):
        """Returns a list of unique person IDs in the dataset.

        Returns:
            list: List of unique person IDs.
        """
        return self._unique_ids
