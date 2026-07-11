from torchvision.datasets.folder import default_loader
import os
import collections
from src.utils.file_path import get_dataset_path

from .abstract import ReIDDataset

__all__ = ['Market1501Dataset']

DATASET_NAME = "market_1501"


def get_data_path():
    """Returns the image and annotation directory paths for different stages.

    Returns:
        dict: A dictionary containing 'train', 'val', 'test', and 'query' paths.
    """
    return {
        "train": get_dataset_path(DATASET_NAME, "bounding_box_train"),
        "val": get_dataset_path(DATASET_NAME, "bounding_box_test"),
        "test": get_dataset_path(DATASET_NAME, "bounding_box_test"),
        "query": get_dataset_path(DATASET_NAME, "query"),
    }


class Market1501Dataset(ReIDDataset):
    """
    Dataset class for Market-1501.
    Expects the standard Market-1501 directory structure.
    """
    name = DATASET_NAME

    def __init__(self, transform, stage, include_junk=False, include_background=False, *args, **kwargs):
        super().__init__(transform, stage, *args, **kwargs)
        self.loader = default_loader
        self.dir = get_data_path()[stage]

        if not os.path.exists(self.dir):
            raise FileNotFoundError(f"Directory {self.dir} not found.")

        self.imgs = [f for f in os.listdir(self.dir) if f.endswith('.jpg')]
        self.imgs.sort()

        self.img_anns = []
        for img_name in self.imgs:
            # Market-1501 filename format: [pid]_c[camid]s[sequence]_[frame]_[junk].jpg
            parts = img_name.split('_')
            pid = int(parts[0])

            if not include_junk and pid == -1:
                continue
            if not include_background and pid == 0:
                continue

            camid = int(parts[1][1])
            self.img_anns.append(
                {"image_path": img_name, "id": pid, "camera": camid})

        self.ids = [ann["id"] for ann in self.img_anns]
        self._unique_ids = sorted(set(self.ids))

        self._id2label = {pid: idx for idx, pid in enumerate(self.unique_ids)}
        id2index = collections.defaultdict(list)
        for idx, pid in enumerate(self.ids):
            id2index[pid].append(idx)
        self._id2index = id2index
        self.cameras = [ann["camera"] for ann in self.img_anns]

    def __len__(self):
        return len(self.img_anns)

    def _getitem(self, index):
        img_ann = self.img_anns[index]
        path = os.path.join(self.dir, img_ann["image_path"])
        id_label = self._id2label[img_ann["id"]]

        image = self.loader(path)

        return image, id_label

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
        return self.img_anns[index]["camera"]

    @property
    def unique_ids(self):
        """Returns a list of unique person IDs in the dataset.

        Returns:
            list: List of unique person IDs.
        """
        return self._unique_ids
