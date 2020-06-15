import open3d
from torch.utils.data import Dataset
import numpy as np
import random
import torch
from torch_geometric.data import Data
import logging
import os
import gc
import glob
from tqdm import tqdm
from base.base_dataset import BaseDataSet
from typing import List


class Matterport(BaseDataSet):
    """Matterport3D dataset.
    """

    color_map = torch.FloatTensor(
        [[0, 0, 0],  # unlabeled
         [174, 199, 232],  # wall
         [152, 223, 138],  # floor
         [31, 119, 180],  # cabinet
         [255, 187, 120],  # bed
         [188, 189, 34],  # chair
         [140, 86, 75],  # sofa
         [255, 152, 150],  # table
         [214, 39, 40],  # door
         [197, 176, 213],  # window
         [148, 103, 189],  # bookshelf
         [196, 156, 148],  # picture
         [23, 190, 207],  # counter
         [247, 182, 210],  # desk
         [219, 219, 141],  # curtain
         [255, 127, 14],  # refrigerator
         [158, 218, 229],  # shower curtain
         [44, 160, 44],  # toilet
         [112, 128, 144],  # sink
         [227, 119, 194],  # bathtub
         [82, 84, 163],
         [23, 46, 10]])  # otherfurn

    classes = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain',
               'toilet', 'sink', 'bathtub', 'otherfurniture', 'tmp']

    def __init__(self, root_dir, start_level, end_level, get_coords=False, benchmark=False, is_train=True,
                 debug_mode=False, transform=None, original_meshes_dir=None, sample_checker=None, include_edges=True):
        super(Matterport, self).__init__(root_dir, start_level, end_level, get_coords, benchmark, is_train,
                                         debug_mode, transform, original_meshes_dir, sample_checker, include_edges)

    @property
    def ignore_classes(self):
        return 0

    def get_full_mesh_label(self, name):
        return torch.load(f"{self._root_dir}/{name}.pt")['labels']

    def _load(self, is_train: bool, benchmark: bool) -> List[str]:
        if is_train:
            file_path = 'dataset/meta/matterport/scenes_train.txt'
        else:
            if not benchmark:
                file_path = 'dataset/meta/matterport/scenes_test.txt'
            else:
                file_path = 'dataset/meta/matterport/scenes_test.txt'

        with open(file_path, 'r') as f:
            set_file_paths = f.read().splitlines()

        return [x.split('/')[-1] for x in glob.glob(f"{self._root_dir}/*.pt")
                if x.split('/')[-1].split('_')[0] in set_file_paths]
