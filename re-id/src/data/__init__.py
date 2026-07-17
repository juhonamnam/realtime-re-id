from torch.utils.data import dataloader
from torch.utils.data.sampler import RandomSampler
from . import seg, reid
from .ReIDRandomSampler import ReIDRandomSampler

__all__ = ['get_dataloaders',
           'get_seg_test_data',
           'get_re_id_test_data']


class SequentialLoader:
    def __init__(self, *loaders: dataloader.DataLoader):
        self.loaders = loaders

    def __iter__(self):
        didx = 0
        for loader in self.loaders:
            for batch in loader:
                yield (didx, batch)
            didx += 1

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)


def get_dataloaders(seg_variant,
                    image_resolution,
                    batch_id,
                    batch_image,
                    val_generator):

    batch_size = batch_id * batch_image

    seg_train_dataset = seg.get_train_dataset(image_resolution,
                                              seg_variant,
                                              random_crop=True,
                                              random_shift=True,
                                              random_padding=True,
                                              random_horizontal_flip=False,
                                              random_erasing=True,
                                              random_resolution_reduce=True)
    re_id_train_datasets = [reid.get_dataset(dataset_name,
                                             image_resolution,
                                             "train",
                                             random_crop=False,
                                             random_shift=True,
                                             random_padding=True,
                                             random_horizontal_flip=False,
                                             random_erasing=True,
                                             random_resolution_reduce=True)
                            for dataset_name in reid.REID_DATASETS.keys()]

    re_id_train_samplers = [ReIDRandomSampler(dataset,
                                              batch_id=batch_id,
                                              batch_image=batch_image)
                            for dataset in re_id_train_datasets]

    re_id_train_loaders = [dataloader.DataLoader(dataset,
                                                 sampler=sampler,
                                                 batch_size=batch_size,
                                                 num_workers=8,
                                                 pin_memory=True)
                           for dataset, sampler in zip(re_id_train_datasets, re_id_train_samplers)]
    re_id_train_loader = SequentialLoader(*re_id_train_loaders)

    seg_train_sampler = RandomSampler(seg_train_dataset,
                                      num_samples=sum(len(sampler) for sampler in re_id_train_samplers))

    seg_train_loader = dataloader.DataLoader(seg_train_dataset,
                                             sampler=seg_train_sampler,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             pin_memory=True)

    seg_val_dataset = seg.get_val_dataset(image_resolution,
                                          seg_variant,
                                          generator=val_generator,
                                          random_crop=True,
                                          random_shift=False,
                                          random_padding=True,
                                          random_horizontal_flip=False,
                                          random_erasing=False,
                                          random_resolution_reduce=True)
    re_id_val_datasets = [reid.get_dataset(dataset_name,
                                           image_resolution,
                                           "val",
                                           generator=val_generator,
                                           random_crop=False,
                                           random_shift=False,
                                           random_padding=True,
                                           random_horizontal_flip=False,
                                           random_erasing=False,
                                           random_resolution_reduce=True)
                          for dataset_name in reid.REID_DATASETS.keys()]

    re_id_val_samplers = [ReIDRandomSampler(dataset,
                                            batch_id=batch_id,
                                            batch_image=batch_image,
                                            generator=val_generator)
                          for dataset in re_id_val_datasets]

    re_id_val_loaders = [dataloader.DataLoader(dataset,
                                               sampler=sampler,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               pin_memory=True)
                         for dataset, sampler in zip(re_id_val_datasets, re_id_val_samplers)]
    re_id_val_loader = SequentialLoader(*re_id_val_loaders)

    seg_val_sampler = RandomSampler(seg_val_dataset,
                                    num_samples=sum(len(sampler)
                                                    for sampler in re_id_val_samplers),
                                    generator=val_generator)

    seg_val_loader = dataloader.DataLoader(seg_val_dataset,
                                           sampler=seg_val_sampler,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           pin_memory=True)

    return (seg_train_loader, re_id_train_loader), (seg_val_loader, re_id_val_loader)


def get_seg_test_data(seg_variant,
                      batch_size,
                      image_resolution):
    seg_dataset = seg.get_test_dataset(image_resolution,
                                       seg_variant,
                                       random_crop=True,
                                       random_shift=False,
                                       random_padding=True,
                                       random_horizontal_flip=False,
                                       random_erasing=False,
                                       random_resolution_reduce=True)
    seg_dataloader = dataloader.DataLoader(seg_dataset,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           pin_memory=True)

    return seg_dataset, seg_dataloader


def get_re_id_test_data(dataset_name,
                        batch_size,
                        image_resolution,
                        random_crop=False,
                        random_shift=False,
                        random_padding=False,
                        random_horizontal_flip=False,
                        random_erasing=False,
                        generator=None):
    ri_dataset = reid.get_dataset(dataset_name,
                                  image_resolution,
                                  "test",
                                  generator=generator,
                                  random_crop=random_crop,
                                  random_shift=random_shift,
                                  random_padding=random_padding,
                                  random_horizontal_flip=random_horizontal_flip,
                                  random_erasing=random_erasing,
                                  random_resolution_reduce=True)
    ri_dataloader = dataloader.DataLoader(ri_dataset,
                                          batch_size=batch_size,
                                          num_workers=8,
                                          pin_memory=True)

    return ri_dataset, ri_dataloader
