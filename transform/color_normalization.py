import torch


class ColorNormalization:
    """Normalize color features to the range [begin, end]."""

    def __init__(self, begin: int = 0 , end: int = 1):
        self._begin = begin
        self._end = end

        # assuming that data is already in range of [0,1]
        self._mins = torch.FloatTensor([0, 0, 0])
        self._maxs = torch.FloatTensor([1, 1, 1])

    def __call__(self, sample):
        sample.x[:, :3] = (self._end - self._begin) * (sample.x[:, :3] - self._mins) / (self._maxs - self._mins) + self._begin

        return sample
