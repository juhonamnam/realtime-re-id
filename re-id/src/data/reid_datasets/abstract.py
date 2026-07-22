from torchvision.datasets.folder import default_loader
from torch.utils.data import dataset
from abc import abstractmethod
import os
import collections


class ReIDDataset(dataset.Dataset):
    """Abstract base class for person re-identification datasets."""

    @abstractmethod
    def get_data_path(self, stage):
        """Returns the image and annotation directory paths for different stages.

        Args:
            stage (str): One of 'train', 'val', 'test', 'query', or 'gallery'.
        Returns:
            dict: A dictionary containing 'train', 'val', 'test', and 'query' paths.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def parse_image_filename(self, filename):
        """Parses the image filename to extract person ID and camera ID.

        Args:
            filename (str): The image filename.
        Returns:
            tuple: A tuple containing (person_id, camera_id).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __init__(self, transform, stage, include_junk=False, include_background=False, **kwargs):
        super().__init__()
        self.transform = transform
        self.stage = stage
        self.loader = default_loader
        self.dir = self.get_data_path(stage)

        if not os.path.exists(self.dir):
            raise FileNotFoundError(f"Directory {self.dir} not found.")

        self.imgs = [f for f in os.listdir(self.dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.imgs.sort()

        self.img_anns = []
        for img_name in self.imgs:
            pid, camid = self.parse_image_filename(img_name)

            if not include_junk and pid == -1:
                continue
            if not include_background and pid == 0:
                continue

            self.img_anns.append(
                {"image_path": img_name, "id": pid, "camera": camid})

        if stage == "train":
            unique_ids = sorted(set(ann["id"] for ann in self.img_anns))
            id2label = {pid: idx for idx, pid in enumerate(unique_ids)}
            for ann in self.img_anns:
                ann["id"] = id2label[ann["id"]]

        self.ids = [ann["id"] for ann in self.img_anns]
        self._unique_ids = sorted(set(self.ids))

        id2index = collections.defaultdict(list)
        for idx, pid in enumerate(self.ids):
            id2index[pid].append(idx)
        self._id2index = id2index
        self.cameras = [ann["camera"] for ann in self.img_anns]

    def __len__(self):
        return len(self.img_anns)

    def __getitem__(self, index):
        """Returns the item at the given index.

        Args:
            index (int or list): Dataset index.

        Returns:
            tuple or list: Data items.
        """
        if isinstance(index, list):
            return [self.__getitem__(i) for i in index]

        img_ann = self.img_anns[index]
        path = os.path.join(self.dir, img_ann["image_path"])
        pid = img_ann["id"]

        image = self.loader(path)
        image = self.transform(image)

        if self.stage in ["query", "gallery"]:
            camera_id = self.get_camera_by_index(index)
            return image, pid, camera_id

        return image, pid

    def get_indices_by_id(self, id):
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
