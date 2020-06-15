class RatioUnlabeled:
    """Checks whether a sample has more than T % labeled vertices in the first hierarchy level."""

    def __init__(self, threshold: float):
        assert 0. <= threshold <= 1.
        self._threshold = threshold

    def __call__(self, sample):
        ratio_unlabeled = (sample.y == 0).sum().float() / sample.y.shape[0]

        return ratio_unlabeled < self._threshold
