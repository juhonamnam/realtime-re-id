from torch.utils.data import sampler
import torch


class ReIDRandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image, generator=None):
        super().__init__()

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id
        self.generator = generator
        self.unique_ids = self.data_source.unique_ids

        self.id_len = len(self.unique_ids) // batch_id * batch_id
        self.id_len = min(self.id_len, len(self.unique_ids))

    def __iter__(self):
        if self.generator:
            indices = torch.randperm(len(self.unique_ids), generator=self.generator, device=self.generator.device)
        else:
            indices = torch.randperm(len(self.unique_ids))

        indices = indices[:self.id_len]

        imgs = []
        for idx in indices:
            _id = self.unique_ids[idx]
            imgs.extend(self._sample(self.data_source.get_indexes_by_id(_id), self.batch_image))
        return iter(imgs)

    def __len__(self):
        return self.id_len * self.batch_image

    def _sample(self, population, k):
        if len(population) < k:
            population = population * k

        if self.generator:
            indices = torch.randperm(len(population), generator=self.generator, device=self.generator.device)
        else:
            indices = torch.randperm(len(population))

        sample = []

        for i in range(k):
            sample.append(population[indices[i]])

        return sample
