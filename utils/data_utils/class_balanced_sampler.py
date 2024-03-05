import torch
from torch.utils.data import Dataset


class ClassBalancedRandomSampler(torch.utils.data.Sampler):
    """
    A custom sampler that sub-samples a given dataset based on class labels. Based on the RandomSampler class
    This is essentially the non-ddp version of ClassBalancedDistributedSampler
    Ref: https://github.com/pytorch/pytorch/blob/abe3c55a6a01c5b625eeb4fc9aab1421a5965cd2/torch/utils/data/sampler.py#L117
    """

    def __init__(self, dataset: Dataset, num_samples_per_class=100, seed: int = 0) -> None:
        self.dataset = dataset
        self.seed = seed
        # Calculate the number of samples
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.num_samples_per_class = num_samples_per_class
        indices = dataset.generate_class_balanced_indices(self.generator,
                                                          num_samples_per_class=num_samples_per_class)
        self.num_samples = len(indices)

    def __iter__(self):
        # Change seed for every function call
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator.manual_seed(seed)
        indices = self.dataset.generate_class_balanced_indices(self.generator, num_samples_per_class=self.num_samples_per_class)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
