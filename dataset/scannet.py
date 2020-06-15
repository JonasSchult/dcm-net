import numpy as np
import random
import torch
from torch_geometric.data import Data
import logging
import os
import gc
import open3d
import glob
from typing import List
from base.base_dataset import BaseDataSet


class ScanNet(BaseDataSet):
    """ScanNet v2 Dataset.
    """
    classes = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain',
               'toilet', 'sink', 'bathtub', 'otherfurniture']

    def __init__(self, root_dir, start_level, end_level, get_coords=False, benchmark=False, is_train=True,
                 debug_mode=False, transform=None, original_meshes_dir=None, sample_checker=None, include_edges=True):
        super(ScanNet, self).__init__(root_dir, start_level, end_level, get_coords, benchmark, is_train,
                                      debug_mode, transform, original_meshes_dir, sample_checker, include_edges)

    @property
    def ignore_classes(self):
        return 0

    @property
    def color_map(self):
        return torch.FloatTensor(
            [[255, 255, 255],  # unlabeled
             [174, 199, 232],  # wall
                [152, 223, 138],  # floor
                [31, 119, 180],  # cabinet
                [255, 187, 120],  # bed
                [188, 189, 34],  # chair
                [140, 86, 75],   # sofa
                [255, 152, 150],  # table
                [214, 39, 40],   # door
                [197, 176, 213],  # window
                [148, 103, 189],  # bookshelf
                [196, 156, 148],  # picture
                [23, 190, 207],  # counter
                [247, 182, 210],  # desk
                [219, 219, 141],  # curtain
                [255, 127, 14],  # refrigerator
                [158, 218, 229],  # shower curtain
                [44, 160, 44],   # toilet
                [112, 128, 144],  # sink
                [227, 119, 194],  # bathtub
                [82, 84, 163]])  # otherfurn

    def get_full_mesh_label(self, name):
        full_label_path = f"{self._original_meshes_dir}/{name.replace('.pt', '')}/{name.replace('.pt', '')}_labels.pt"
        return torch.load(full_label_path)

    def get_full_mesh_size(self, name):
        full_label_path = f"{self._original_meshes_dir}/{name}/{name}_vh_clean_2_size.txt"

        with open(full_label_path, 'r') as read_file:
            out = int(read_file.readline())

        return out

    def get_mesh(self, mesh_name):
        mesh_name = mesh_name.replace('.pt', '')
        mesh_rgb_path = f"{self._original_meshes_dir}/{mesh_name}/{mesh_name}_vh_clean_2.ply"
        return open3d.read_triangle_mesh(mesh_rgb_path)

    def _load(self, is_train: bool, benchmark: bool) -> List[str]:
        if is_train:
            file_path = 'dataset/meta/scannet/scannetv2_train.txt'
        else:
            if not benchmark:
                file_path = 'dataset/meta/scannet/scannetv2_val.txt'
            else:
                file_path = 'dataset/meta/scannet/scannetv2_test.txt'

        with open(file_path, 'r') as f:
            set_file_paths = f.read().splitlines()

        if is_train:
            return [x.split('/')[-1] for x in glob.glob(f"{self._root_dir}/*.pt")
                    if x.split('/')[-1].replace('.pt', '').rsplit('_', 1)[0] in set_file_paths]
        else:
            return [x.split('/')[-1] for x in glob.glob(f"{self._root_dir}/*.pt")
                    if x.split('/')[-1].replace('.pt', '') in set_file_paths]
