from torch.utils.data import dataloader
from torch.utils.data.sampler import RandomSampler
from . import seg, reid
from .ReIDRandomSampler import ReIDRandomSampler

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

def get_bb_dataloaders(seg_variant,
                       image_resolution,
                       batch_id,
                       batch_image,
                       val_generator):

    batch_size = batch_id * batch_image

    seg_train_dataset = seg.get_train_dataset(image_resolution,
                                              seg_variant,
                                              random_crop=True,
                                              random_horizontal_flip=True,
                                              random_resolution_reduce=True)
    re_id_train_dataset = reid.get_dataset(image_resolution,
                                           "train",
                                           random_crop=True,
                                           random_horizontal_flip=True,
                                           random_erasing=False,
                                           random_resolution_reduce=True)

    re_id_train_sampler = ReIDRandomSampler(re_id_train_dataset,
                                            batch_id=batch_id,
                                            batch_image=batch_image)

    seg_train_sampler = RandomSampler(seg_train_dataset,
                                      num_samples=len(re_id_train_sampler))

    seg_train_loader = dataloader.DataLoader(seg_train_dataset,
                                             sampler=seg_train_sampler,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             pin_memory=True)

    re_id_train_loader = dataloader.DataLoader(re_id_train_dataset,
                                               sampler=re_id_train_sampler,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               pin_memory=True)

    seg_val_dataset = seg.get_val_dataset(image_resolution,
                                          seg_variant,
                                          generator=val_generator,
                                          random_crop=True,
                                          random_horizontal_flip=True,
                                          random_resolution_reduce=True)
    re_id_val_dataset = reid.get_dataset(image_resolution,
                                         "val",
                                         generator=val_generator,
                                         random_crop=True,
                                         random_horizontal_flip=True,
                                         random_erasing=False,
                                         random_resolution_reduce=True)
    re_id_val_sampler = ReIDRandomSampler(re_id_val_dataset,
                                           batch_id=batch_id,
                                           batch_image=batch_image,
                                           generator=val_generator)

    seg_val_sampler = RandomSampler(seg_val_dataset,
                                    num_samples=len(re_id_val_sampler),
                                    generator=val_generator)

    seg_val_loader = dataloader.DataLoader(seg_val_dataset,
                                           sampler=seg_val_sampler,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           pin_memory=True)

    re_id_val_loader = dataloader.DataLoader(re_id_val_dataset,
                                             sampler=re_id_val_sampler,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             pin_memory=True)

    return (seg_train_loader, re_id_train_loader), (seg_val_loader, re_id_val_loader)


def get_seg_dataloaders(variant,
                        image_resolution,
                        batch_size,
                        val_generator=None):
    train_dataset = seg.get_train_dataset(image_resolution,
                                          variant,
                                          random_crop=True,
                                          random_horizontal_flip=True,
                                          random_resolution_reduce=True)

    train_sampler = RandomSampler(train_dataset)
    train_loader = dataloader.DataLoader(train_dataset,
                                         sampler=train_sampler,
                                         batch_size=batch_size,
                                         num_workers=8,
                                         pin_memory=True)

    val_dataset = seg.get_val_dataset(image_resolution,
                                      variant,
                                      generator=val_generator,
                                      random_crop=True,
                                      random_horizontal_flip=True,
                                      random_resolution_reduce=True)

    val_loader = dataloader.DataLoader(val_dataset,
                                       batch_size=batch_size,
                                       num_workers=8,
                                       pin_memory=True)

    return train_loader, val_loader


def get_re_id_dataloaders(image_resolution,
                          batch_id,
                          batch_image,
                          val_generator=None):

    batch_size = batch_id * batch_image

    train_dataset = reid.get_dataset(image_resolution,
                                     "train",
                                     random_crop=True,
                                     random_horizontal_flip=True,
                                     random_erasing=False,
                                     random_resolution_reduce=True)

    train_sampler = ReIDRandomSampler(train_dataset,
                                      batch_id=batch_id,
                                      batch_image=batch_image)

    train_loader = dataloader.DataLoader(train_dataset,
                                         sampler=train_sampler,
                                         batch_size=batch_size,
                                         num_workers=8,
                                         pin_memory=True)

    val_dataset = reid.get_dataset(image_resolution,
                                   "val",
                                   generator=val_generator,
                                   random_crop=True,
                                   random_horizontal_flip=True,
                                   random_erasing=False,
                                   random_resolution_reduce=True)

    val_sampler = ReIDRandomSampler(val_dataset,
                                    batch_id=batch_id,
                                    batch_image=batch_image,
                                    generator=val_generator)

    val_loader = dataloader.DataLoader(val_dataset,
                                         sampler=val_sampler,
                                         batch_size=batch_size,
                                         num_workers=8,
                                         pin_memory=True)

    return train_loader, val_loader


def get_seg_test_data(seg_variant,
                      batch_size,
                      image_resolution):
    seg_dataset = seg.get_test_dataset(image_resolution,
                                       seg_variant,
                                       random_crop=True,
                                       random_horizontal_flip=True,
                                       random_resolution_reduce=True)
    seg_dataloader = dataloader.DataLoader(seg_dataset,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           pin_memory=True)

    return seg_dataset, seg_dataloader

def get_re_id_test_data(batch_size,
                        image_resolution,
                        random_crop=True,
                        random_horizontal_flip=True,
                        random_erasing=False,
                        generator=None):
    ri_dataset = reid.get_dataset(image_resolution,
                                  "test",
                                  generator=generator,
                                  random_crop=random_crop,
                                  random_horizontal_flip=random_horizontal_flip,
                                  random_erasing=random_erasing,
                                  random_resolution_reduce=True)
    ri_dataloader = dataloader.DataLoader(ri_dataset,
                                          batch_size=batch_size,
                                          num_workers=8,
                                          pin_memory=True)

    return ri_dataset, ri_dataloader
