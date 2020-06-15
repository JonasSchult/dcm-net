import torch


class RandomLinearTransformation:
    """Apply random linear transformation on all vertex positions in all hierarchy levels."""

    def __init__(self, flip: bool = True, pertubation_factor: float = 0.1):
        """Initialize random linear transformation augmentation

        Keyword Arguments:
            flip {bool} -- flip along z axis (default: {True})
            pertubation_factor {float} -- how much noise should be added? (default: {0.1})
        """
        self._flip = flip
        self._pertubation_factor = pertubation_factor

    def __call__(self, sample):
        m = torch.eye(3) + torch.randn(3, 3) * self._pertubation_factor

        if self._flip:
            m[0, 0] *= - 1

        pos_idx = [a for a in dir(sample) if 'pos' in a]

        for pos_id in pos_idx:
            sample[pos_id] = sample[pos_id] @ m

        return sample
