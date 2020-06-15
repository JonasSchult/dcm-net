class EnoughPoints:
    """Checks whether a sample has at least T vertices in the final hierachy level."""

    def __init__(self, threshold: int):
        assert threshold >= 0
        self._threshold = threshold

    def __call__(self, sample):
        return self._threshold < sample.num_vertices[-1]
