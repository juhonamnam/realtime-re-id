import torch.nn as nn
from torchvision import transforms

from .transforms.RandomShift import RandomShift
from .transforms.RandomErasing import RandomErasing
from .transforms.RandomHorizontalFlip import RandomHorizontalFlip
from .transforms.RandomCrop import RandomCrop
from .transforms.LetterboxPad import LetterboxPad
from .transforms.RandomResolutionReduce import RandomResolutionReduce

from .reid_datasets.market_1501 import Market1501Dataset
from .reid_datasets.nia import NIADataset
from .reid_datasets.duke_mtmc_reid import DukeMTMCReIDDataset


__all__ = ['get_dataset', 'REID_DATASETS']

REID_DATASETS = {
    "market1501": Market1501Dataset,
    "duke": DukeMTMCReIDDataset,
    "nia": NIADataset,
}


class ReIDTransform(nn.Module):

    def __init__(self, image_resolution, generator, random_crop,
                 random_shift, random_padding, random_horizontal_flip,
                 random_erasing, random_resolution_reduce):
        super().__init__()

        t = []

        if random_crop:
            t.append(RandomCrop(generator=generator))

        t.append(transforms.ToTensor())

        if random_shift:
            t.append(RandomShift(generator=generator))

        if random_erasing:
            t.append(RandomErasing(generator=generator))

        t.append(LetterboxPad(size=image_resolution,
                 random_padding=random_padding, generator=generator))

        if random_resolution_reduce:
            t.append(RandomResolutionReduce(
                target_size=image_resolution, generator=generator))

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


def get_dataset(dataset_name,
                image_resolution,
                data_stage,
                generator=None,
                no_augment=False,
                random_crop=False,
                random_shift=False,
                random_padding=True,
                random_horizontal_flip=False,
                random_erasing=False,
                random_resolution_reduce=True,
                iterate_camera_id=False):

    if no_augment:
        random_crop = False
        random_shift = False
        random_padding = False
        random_horizontal_flip = False
        random_erasing = False
        random_resolution_reduce = False

    transform = ReIDTransform(image_resolution=image_resolution,
                              generator=generator,
                              random_crop=random_crop,
                              random_shift=random_shift,
                              random_padding=random_padding,
                              random_horizontal_flip=random_horizontal_flip,
                              random_erasing=random_erasing,
                              random_resolution_reduce=random_resolution_reduce)

    dataset = REID_DATASETS.get(dataset_name.lower())
    if dataset is None:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets: {list(REID_DATASETS.keys())}")

    return dataset(transform=transform,
                   stage=data_stage,
                   iterate_camera_id=iterate_camera_id)
