import torch
from torch.utils.data import Dataset
from typing import Optional
import math
import torch.distributed as dist


class ClassBalancedDistributedSampler(torch.utils.data.Sampler):
    """
    A custom sampler that sub-samples a given dataset based on class labels. Based on the DistributedSampler class
    Ref: https://github.com/pytorch/pytorch/blob/04c1df651aa58bea50977f4efcf19b09ce27cefd/torch/utils/data/distributed.py#L13
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False, num_samples_per_class=100) -> None:

        if not shuffle:
            raise ValueError("ClassBalancedDatasetSubSampler requires shuffling, otherwise use DistributedSampler")

        # Check if the dataset has a generate_class_balanced_indices method
        if not hasattr(dataset, 'generate_class_balanced_indices'):
            raise ValueError("Dataset does not have a generate_class_balanced_indices method")

        self.shuffle = shuffle
        self.seed = seed
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # Calculate the number of samples
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.num_samples_per_class = num_samples_per_class
        indices = dataset.generate_class_balanced_indices(torch.Generator(),
                                                          num_samples_per_class=num_samples_per_class)
        dataset_size = len(indices)

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (dataset_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(dataset_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch and seed, here shuffle is assumed to be True
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.dataset.generate_class_balanced_indices(g, num_samples_per_class=self.num_samples_per_class)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
