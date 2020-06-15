import torch
import glob
import logging
import os
import gc
import numpy as np
import open3d
from typing import List
from torch_geometric.data import Data
from sklearn.neighbors import BallTree
from base.base_dataset import BaseDataSet
from torch.utils.data import Dataset


class S3DIS(BaseDataSet):
    """Stanford 3D Large-Scale Indoor Scenes Dataset.
    """

    color_map = torch.FloatTensor(
        [[0, 255, 0],
         [0, 0, 255],
         [0, 255, 255],
         [255, 255, 0],
         [255, 0, 255],
         [100, 100, 255],
         [200, 200, 100],
         [170, 120, 200],
         [255, 0, 0],
         [200, 100, 100],
         [10, 200, 100],
         [200, 200, 200],
         [50, 50, 50]])

    classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
               'board', 'clutter']

    def __init__(self, root_dir, start_level, end_level, get_coords=False, benchmark=False, is_train=True,
                 debug_mode=False, transform=None, original_meshes_dir=None, sample_checker=None, include_edges=True, test_area='Area_5'):
        self._test_area = test_area
        super(S3DIS, self).__init__(root_dir, start_level, end_level, get_coords, benchmark, is_train,
                                    debug_mode, transform, original_meshes_dir, sample_checker, include_edges)

    @property
    def ignore_classes(self):
        return None

    def get_level_0(self, name):
        return torch.load(f"{self._root_dir}/{name}")['vertices'][0]

    def get_level_0_labels(self, name):
        name = name + '_full'
        prefix = f"{name}/{name.split('/')[-1]}"
        return torch.load(f"{prefix}_labels_0.pt")

    def get_full_mesh_size(self, name):
        full_label_path = f"{self._original_meshes_dir}/{name}/{name}_size.txt"

        with open(full_label_path, 'r') as read_file:
            out = int(read_file.readline())

        return out

    def get_full_mesh_label(self, name):
        full_label_path = f"{self._original_meshes_dir}/{name.split('_',2)[-1].replace('.pt', '')}/{name.split('_',2)[-1].replace('.pt', '')}.labels.npy"
        return torch.from_numpy(np.load(full_label_path))

    def get_gt_pointcloud(self, name):
        full_label_path = f"{self._original_meshes_dir}/{name.split('_', 2)[-1].replace('.pt', '')}_gt_wo_colors.pt"
        return torch.load(full_label_path)

    def _load(self, is_train: bool, benchmark: bool) -> List[str]:
        if is_train:
            return [x.split('/')[-1] for x in glob.glob(f"{self._root_dir}/*.pt")
                    if not x.split('/')[-1].startswith(self._test_area)]

        return [x.split('/')[-1] for x in glob.glob(f"{self._root_dir}/*.pt")
                if x.split('/')[-1].startswith(self._test_area)]
