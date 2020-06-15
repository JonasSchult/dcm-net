import torch


class MoveToOrigin:
    """Before passing the crop to the network we move it to the origin, e.g. its center is at the origin"""

    def __init__(self):
        pass

    def __call__(self, sample):
        pos_idx = [a for a in dir(sample) if 'pos' in a]

        # center is calculated for the first hierarchy level
        middle = (sample['pos'][:, :3].max(dim=0)[0] + sample['pos'][:, :3].min(dim=0)[0]) / 2

        for pos_id in pos_idx:
            sample[pos_id] -= middle

        return sample
