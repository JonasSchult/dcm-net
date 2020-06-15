"""We train on crops of full rooms. This scripts creates a database with such training crops.
"""

import os
from tqdm import tqdm
import argparse
from argparse import RawTextHelpFormatter
import glob
import numpy as np
from sklearn.neighbors import BallTree
import open3d
import torch
import shutil
from termcolor import colored
import time
from typing import Tuple
import sys
sys.path.append("..")
from clear_folder import clear_folder
from pretty_print import pretty_print_arguments


def get_sampling_positions(original_vertices: np.ndarray, stride: float) -> Tuple[np.ndarray, np.ndarray]:
    """returns center points of uniform sampling with certain stride length along the ground plane

    Arguments:
        original_vertices {np.ndarray} -- 3D coordinates of vertices
        stride {float} -- distance between center points of sampling

    Returns:
        Tuple[np.ndarray, np.ndarray] -- returns sampling positions in x and y direction (always full height!)
    """
    mins_xyz = original_vertices[:, :3].min(axis=0)
    maxs_xyz = original_vertices[:, :3].max(axis=0)

    sampling_positions_x = np.arange(mins_xyz[0], maxs_xyz[0], stride)
    offset_x = (maxs_xyz[0] - sampling_positions_x[-1]) / 2
    sampling_positions_x = sampling_positions_x + offset_x

    sampling_positions_y = np.arange(mins_xyz[1], maxs_xyz[1], stride)
    offset_y = (maxs_xyz[1] - sampling_positions_y[-1]) / 2
    sampling_positions_y = sampling_positions_y + offset_y

    return sampling_positions_x, sampling_positions_y


def process_frame(file_path):
    saved_tensors = torch.load(file_path)
    vertices = [x.numpy() for x in saved_tensors['vertices']]
    edges = [x.numpy() for x in saved_tensors['edges']]
    traces = [x.numpy() for x in saved_tensors['traces']]

    if 'labels' in saved_tensors:
        is_train = True
        labels = saved_tensors['labels'].numpy()
    else:
        is_train = False

    del saved_tensors

    block_size = args.block_size
    stride = args.stride

    sampling_positions_x, sampling_positions_y = get_sampling_positions(
        vertices[0], stride)

    block_counter = 0
    for x_pos in sampling_positions_x:
        for y_pos in sampling_positions_y:
            try:
                # find borders of current crop
                min_bound = np.array(
                    [x_pos - block_size / 2, y_pos - block_size / 2, float("-inf")])
                max_bound = np.array(
                    [x_pos + block_size / 2, y_pos + block_size / 2, float("inf")])

                filters = []
                block_coords = []
                block_edges = []
                block_traces = []

                for level, vert in enumerate(vertices):
                    # crop in each hierarchy level accordingly
                    min_check = (vert[:, :3] >= min_bound).sum(axis=1) == 3
                    max_check = (vert[:, :3] <= max_bound).sum(axis=1) == 3
                    filters.append(min_check * max_check)

                    # take care that edges do not point outside of the current crop
                    block_edges.append(edges[level][np.isin(edges[level],
                                                            np.argwhere(filters[-1]).flatten()).sum(axis=1) == 2])

                    new_filter = np.ndarray(filters[-1].shape, dtype=np.bool)
                    new_filter[:] = False
                    new_filter[np.unique(block_edges[-1])] = True
                    filters[-1] = new_filter
                    block_coords.append(vert[filters[-1]])

                    if level == 0 and is_train:
                        pooled_levels = np.zeros(
                            (traces[0].max() + 1, labels.max() + 1), dtype=np.int)
                        for i in range(len(labels)):
                            pooled_levels[traces[0][i]][labels[i]] += 1

                        pooled_levels = np.argmax(pooled_levels, axis=1)
                        block_labels = pooled_levels[filters[-1]]

                    # map from old indices to new ones
                    block_edges[-1] = np.unique(block_edges[-1],
                                                return_inverse=True)[1].reshape(-1, 2)

                if min([len(coords) for coords in block_coords]) == 0 or block_coords[-1].shape[0] < 50:
                    print(colored('Warning: Too less points for crop -> reject', 'red'))
                    block_counter += 1
                    continue

                for level in range(len(traces)-1):
                    block_traces.append(traces[level+1][filters[level]])
                    nonreduced_mappings = vertices[level+1][block_traces[-1]]

                    min_check = (nonreduced_mappings[:, :3] >= min_bound).sum(
                        axis=1) == 3
                    max_check = (nonreduced_mappings[:, :3] <= max_bound).sum(
                        axis=1) == 3

                    block_traces[-1][~(min_check * max_check)
                                     ] = block_traces[-1][min_check * max_check][0]
                    save = vertices[level+1][block_traces[-1]]

                    # if trace points to a representative vertex outside of current crop,
                    # redirect it to the nearest neighbor in the subsequent graph level
                    ball_tree = BallTree(block_coords[level+1][:, :3])
                    d, ind = ball_tree.query(save[:, :3], k=1)
                    block_traces[-1][d.flatten() >
                                     0] = block_traces[-1][d.flatten() == 0][0]
                    block_traces[-1] = ind.flatten()

                    if block_coords[level][np.logical_or(d.flatten() > 0, ~(min_check * max_check))].shape[0] != 0:
                        dist2, ind2 = ball_tree.query(block_coords[level][np.logical_or(
                            d.flatten() > 0, ~(min_check * max_check))][:, :3], k=1)
                        block_traces[-1][np.logical_or(d.flatten() > 0, ~(
                            min_check * max_check))] = ind2.flatten()

                    if np.unique(block_traces[-1], axis=0).shape[0] < block_coords[level+1].shape[0]:
                        # representation vertex exists with no predecessor
                        # find nearest neighbor and trace to this point -> quick and dirty fix;
                        # better delete it but then also keep care of edges, following levels and trace map ids
                        missing_ids_in_next_level = np.array(
                            list(set(range(block_coords[level+1].shape[0])) - set(block_traces[-1])))
                        ball_tree = BallTree(block_coords[level][:, :3])
                        prev_level_ids = ball_tree.query(block_coords[level + 1][missing_ids_in_next_level][:, :3],
                                                         k=block_coords[level][:, :3].shape[0])[1]

                        for v_id in range(prev_level_ids.shape[0]):
                            success = False
                            for neighbor in prev_level_ids[v_id]:
                                if (block_traces[-1] == block_traces[-1][neighbor]).sum() > 1:
                                    success = True
                                    block_traces[-1][neighbor] = missing_ids_in_next_level[v_id]
                                    break
                            if not success:
                                raise ValueError('CROP GRAPH LEVEL ERROR')

                        if np.unique(block_traces[-1], axis=0).shape[0] != block_coords[level+1].shape[0]:
                            raise ValueError('CROP GRAPH LEVEL ERROR')

                    elif np.unique(block_traces[-1], axis=0).shape[0] > block_coords[level+1].shape[0]:
                        raise ValueError('CROP GRAPH LEVEL ERROR')

                pt_data = {}

                block_vertices = [torch.from_numpy(block_coords[i]).float() for i in
                                  range(len(block_coords))]

                block_edges = [torch.from_numpy(block_edges[i]).long() for i in
                               range(len(block_edges))]

                block_traces = [torch.from_numpy(block_traces[i]).long() for i in
                                range(len(block_traces))]

                pt_data['vertices'] = block_vertices
                pt_data['edges'] = block_edges
                pt_data['traces'] = block_traces

                if is_train:
                    block_labels = torch.from_numpy(block_labels).long()
                    pt_data['labels'] = block_labels

                torch.save(
                    pt_data, f"{args.out_path}{file_path.split('/')[-1].replace('.pt', '')}_{block_counter}.pt")

            except Exception as e:
                print(e)
            finally:
                block_counter += 1


def main(args):
    file_paths = sorted([x for x in glob.glob(f"{args.in_path}/*.pt")])

    if not os.path.exists(os.path.dirname(args.out_path)):
        os.makedirs(os.path.dirname(args.out_path))

    process_frame(file_paths[args.number])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description="Create crops from graph hierarchies")

    parser.add_argument('--in_path', type=str, required=True,
                        help='path to the root directory of folders containing the original dataset')

    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the directory where the crops and tracing files should be stored')

    parser.add_argument('--block_size', const=5., default=5., type=float, nargs='?',
                        help='block size in meter')

    parser.add_argument('--stride', const=1., default=1., type=float, nargs='?',
                        help='stride between center points')

    parser.add_argument('--number', const=-1, default=-1, type=int, nargs='?',
                        help='number of task id (used for parallization with SLURM)')

    args = parser.parse_args()
    pretty_print_arguments(args)
    open3d.set_verbosity_level(open3d.VerbosityLevel.Error)
    main(args)
