import torch
import math


class RandomRotation:
    """Random rotations along height axis."""    

    def __init__(self):
        pass

    def __call__(self, sample):
        theta = torch.rand(1) * 2 * math.pi

        rot_matrix = torch.FloatTensor([[math.cos(theta), math.sin(theta), 0],
                                        [-math.sin(theta), math.cos(theta), 0],
                                        [0, 0, 1]])

        pos_idx = [a for a in dir(sample) if 'pos' in a]

        for pos_id in pos_idx:
            sample[pos_id] = sample[pos_id] @ rot_matrix

        return sample
