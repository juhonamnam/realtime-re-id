from torch.utils.data import sampler
import torch


class ReIDRandomSampler(sampler.Sampler):
    """Random sampler for Re-ID datasets that ensures each batch contains a balanced number of IDs and images.

    Attributes:
        data_source (Dataset): The dataset to sample from.
        batch_id (int): Number of unique person IDs per batch.
        batch_image (int): Number of images per person ID per batch.
        generator (torch.Generator, optional): Random number generator.
        unique_ids (list): List of unique person IDs in the data source.
        id_len (int): Total number of unique IDs to sample from, adjusted for batch size.
    """
    def __init__(self, data_source, batch_id, batch_image, generator=None):
        """Initializes ReIDRandomSampler.

        Args:
            data_source (Dataset): The dataset instance.
            batch_id (int): Desired number of IDs per batch.
            batch_image (int): Desired number of images per ID.
            generator (torch.Generator, optional): Random generator. Defaults to None.
        """
        super().__init__()

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id
        self.generator = generator
        self.unique_ids = self.data_source.unique_ids

        self.id_len = len(self.unique_ids) // batch_id * batch_id
        self.id_len = min(self.id_len, len(self.unique_ids))

    def __iter__(self):
        """Iterates over the sampled indices.

        Returns:
            iterator: An iterator over indices.
        """
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
        """Returns the total number of samples in one iteration.

        Returns:
            int: Total samples.
        """
        return self.id_len * self.batch_image

    def _sample(self, population, k):
        """Samples k indices from a population.

        Args:
            population (list[int]): List of indices for a specific ID.
            k (int): Number of samples to draw.

        Returns:
            list[int]: Sampled indices.
        """
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
