import torch
from typing import List

class CoordsNormalization:
    """Normalize coordinate features of each hierarchy level to the range of max_sizes."""

    def __init__(self, max_sizes: List[int]):
        self.max_sizes = torch.FloatTensor(max_sizes)

    def __call__(self, sample):
        pos_idx = [a for a in dir(sample) if 'pos' in a]

        for pos_id in pos_idx:
            sample[pos_id] = sample[pos_id] / self.max_sizes

        return sample
