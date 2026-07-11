from torch.utils.data import dataset
from abc import abstractmethod


class ReIDDataset(dataset.Dataset):
    """Abstract base class for person re-identification datasets."""

    @abstractmethod
    def __init__(self, transform, stage, return_camera_id=False):
        self.transform = transform
        self.return_camera_id = return_camera_id

    @abstractmethod
    def get_indices_by_id(self, id):
        pass

    @abstractmethod
    def get_camera_by_index(self, index):
        pass

    @abstractmethod
    def _getitem(self, index) -> tuple:
        pass

    def __getitem__(self, index):
        """Returns the item at the given index.

        Args:
            index (int or list): Dataset index.

        Returns:
            tuple or list: Data items.
        """
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]

        image, id_label = self._getitem(index)

        image = self.transform(image)

        if self.return_camera_id:
            camera_id = self.get_camera_by_index(index)
            return image, id_label, camera_id

        return image, id_label

    @property
    @abstractmethod
    def unique_ids(self):
        pass
