from torchvision.datasets.folder import default_loader
from src.utils.file_path import get_dataset_path

from .abstract import ReIDDataset

__all__ = ['NIADataset']

DATASET_NAME = "nia"


def get_data_path():
    return {
        "train": get_dataset_path(DATASET_NAME, "train", "images"),
        "val": get_dataset_path(DATASET_NAME, "val", "images"),
        "test": get_dataset_path(DATASET_NAME, "val", "images"),
    }


class NIADataset(ReIDDataset):
    """
    Dataset class for NIA dataset.
    Expects the standard NIA directory structure.
    """

    def get_data_path(self, stage):
        return get_data_path()[stage]

    def parse_image_filename(self, filename):
        # NIA filename format: [IN/OUT]_H[pid]_[case]_[camid]_[frame].png
        parts = filename.split('_')
        pid = int(parts[1][1:])
        camid = int(parts[3])

        return pid, camid
