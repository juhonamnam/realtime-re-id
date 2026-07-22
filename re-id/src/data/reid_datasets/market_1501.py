from torchvision.datasets.folder import default_loader
from src.utils.file_path import get_dataset_path

from .abstract import ReIDDataset

__all__ = ['Market1501Dataset']

DATASET_NAME = "market_1501"


def get_data_path():
    return {
        "train": get_dataset_path(DATASET_NAME, "bounding_box_train"),
        "val": get_dataset_path(DATASET_NAME, "bounding_box_test"),
        "test": get_dataset_path(DATASET_NAME, "bounding_box_test"),
        "query": get_dataset_path(DATASET_NAME, "query"),
        "gallery": get_dataset_path(DATASET_NAME, "bounding_box_test"),
    }


class Market1501Dataset(ReIDDataset):
    """
    Dataset class for Market-1501.
    Expects the standard Market-1501 directory structure.
    """

    def get_data_path(self, stage):
        return get_data_path()[stage]

    def parse_image_filename(self, filename):
        # Market-1501 filename format: [pid]_c[camid]s[sequence]_[frame]_[junk].jpg
        parts = filename.split('_')
        pid = int(parts[0])
        camid = int(parts[1][1])

        return pid, camid
