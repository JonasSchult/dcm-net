""" Create hierarchy of graph levels by leveragin mesh simplification algorithms.
"""

import argparse
from argparse import RawTextHelpFormatter
import csv
import glob
import os
import shutil
from collections import defaultdict
from typing import List
import numpy as np
import open3d
import pandas as pd
import torch
from plyfile import PlyData
from sklearn.neighbors import BallTree

import sys
sys.path.append("..")
from pretty_print import pretty_print_arguments
from clear_folder import clear_folder

SCANNET_CLASS_REMAP = np.zeros(41)

SCANNET_CLASS_REMAP[1] = 1
SCANNET_CLASS_REMAP[2] = 2
SCANNET_CLASS_REMAP[3] = 3
SCANNET_CLASS_REMAP[4] = 4
SCANNET_CLASS_REMAP[5] = 5
SCANNET_CLASS_REMAP[6] = 6
SCANNET_CLASS_REMAP[7] = 7
SCANNET_CLASS_REMAP[8] = 8
SCANNET_CLASS_REMAP[9] = 9
SCANNET_CLASS_REMAP[10] = 10
SCANNET_CLASS_REMAP[11] = 11
SCANNET_CLASS_REMAP[12] = 12
SCANNET_CLASS_REMAP[14] = 13
SCANNET_CLASS_REMAP[16] = 14
SCANNET_CLASS_REMAP[24] = 15
SCANNET_CLASS_REMAP[28] = 16
SCANNET_CLASS_REMAP[33] = 17
SCANNET_CLASS_REMAP[34] = 18
SCANNET_CLASS_REMAP[36] = 19
SCANNET_CLASS_REMAP[39] = 20

MATTERPORT_CLASS_REMAP = np.zeros(41)

MATTERPORT_CLASS_REMAP[1] = 1
MATTERPORT_CLASS_REMAP[2] = 2
MATTERPORT_CLASS_REMAP[3] = 3
MATTERPORT_CLASS_REMAP[4] = 4
MATTERPORT_CLASS_REMAP[5] = 5
MATTERPORT_CLASS_REMAP[6] = 6
MATTERPORT_CLASS_REMAP[7] = 7
MATTERPORT_CLASS_REMAP[8] = 8
MATTERPORT_CLASS_REMAP[9] = 9
MATTERPORT_CLASS_REMAP[10] = 10
MATTERPORT_CLASS_REMAP[11] = 11
MATTERPORT_CLASS_REMAP[12] = 12
MATTERPORT_CLASS_REMAP[14] = 13
MATTERPORT_CLASS_REMAP[16] = 14
MATTERPORT_CLASS_REMAP[22] = 21  # DIFFERENCE TO SCANNET!
MATTERPORT_CLASS_REMAP[24] = 15
MATTERPORT_CLASS_REMAP[28] = 16
MATTERPORT_CLASS_REMAP[33] = 17
MATTERPORT_CLASS_REMAP[34] = 18
MATTERPORT_CLASS_REMAP[36] = 19
MATTERPORT_CLASS_REMAP[39] = 20

# scannet classes + ceiling
MATTERPORT_ALLOWED_NYU_CLASSES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 22, 24, 28, 33, 34, 36, 39]


class QEMError(Exception):
    pass


def arg_check_levels(value):
    val = int(value)
    if val < 1:
        raise argparse.ArgumentTypeError(
            "The number of levels has to exceed 0. It is %s, though." % value)
    return val


def arg_check_percentage(value):
    val = int(value)
    if val <= 0 or val >= 100:
        raise argparse.ArgumentTypeError(
            "%s is an invalid percentage value" % value)
    return val


def get_color_and_labels(original_vertices: np.ndarray, representative_vertices: np.ndarray) -> List[np.ndarray]:
    """find nearest neighbor in Euclidean space to interpolate color and label information to vertices in simplified mesh.

    Arguments:
        original_vertices {np.ndarray} -- vertex positions in original mesh
        representative_vertices {np.ndarray} -- vertex positions in simplified mesh

    Returns:
        List[np.ndarray] -- list of arrays containing RGB color and label information
    """
    ball_tree = BallTree(original_vertices[:, :3])

    return_colors_labels = []

    for coords in representative_vertices:
        _, ind = ball_tree.query(coords, k=1)
        return_colors_labels.append(original_vertices[ind.flatten()][:, 3:])

    return return_colors_labels


def edges_from_faces(faces):
    edges = defaultdict(set)
    for i in range(len(faces)):
        edges[faces[i, 0]].update(faces[i, (1, 2)])
        edges[faces[i, 1]].update(faces[i, (0, 2)])
        edges[faces[i, 2]].update(faces[i, (0, 1)])

    edge_list = []

    for vertex_id in range(len(edges)):
        connected_vertices = edges[vertex_id]
        edge_list.append(list(connected_vertices))

    return edge_list


def csv2npy(in_file_path, old_vertices, new_vertices):
    old_ball_tree = BallTree(old_vertices[:, :3])
    new_ball_tree = BallTree(new_vertices[:, :3])

    trace = []

    new2old = {}
    old_nodes_set = set()
    new_nodes_set = set()
    num_nodes = 0
    with open(in_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            new_coords = [float(r) for r in row[:3]]
            num_traces = len(row) // 3 - 1

            new_id = new_ball_tree.query(
                np.array(new_coords).reshape(1, -1), k=1)[1].flatten()[0]
            if not new2old.get(new_id):
                new2old[new_id] = []
            else:
                raise QEMError('GRAPH LEVEL GENERATION ERROR')

            new_nodes_set.add(new_id)

            for i in range(num_traces):
                old_coords = [float(r) for r in row[3+3*i:6+3*i]]
                old_id = old_ball_tree.query(
                    np.array(old_coords).reshape(1, -1), k=1)[1].flatten()[0]
                new2old[new_id].append(old_id)
                old_nodes_set.add(old_id)

    all_old_ids = list(
        set([i for i in range(old_vertices.shape[0])]) - old_nodes_set)
    if np.array(old_vertices[all_old_ids]).shape[0] != 0:
        new_ids = new_ball_tree.query(
            np.array(old_vertices[all_old_ids]), k=1)[1].flatten()
        new_nodes_set.update(new_ids.tolist())
        old_nodes_set.update(all_old_ids)

        for id, new_id in enumerate(new_ids):
            if not new2old.get(new_id):
                new2old[new_id] = []
            new2old[new_id].append(all_old_ids[id])

    assert new_vertices.shape[0] == len(new_nodes_set)
    assert old_vertices.shape[0] == len(old_nodes_set)

    reverse_trace = np.empty((len(old_nodes_set)), dtype=np.int32)
    reverse_trace.fill(-1)

    for new_id, old_ids in new2old.items():
        for old_id in old_ids:
            assert reverse_trace[old_id] == -1
            reverse_trace[old_id] = new_id

    return reverse_trace


def vertex_clustering(coords: np.ndarray, edges: List[List[int]], voxel_size: float):
    """Vertex Clustering as proposed in
    'Multi-Resolution 3D Approximations for Rendering Complex Scenes' by Rossignac et. al.

    Arguments:
        coords {np.ndarray} -- 3D coordinates of vertices
        edges {List[List[int]]} -- adjacency of vertices encoded as list of lists
        voxel_size {float} -- cell size for aggregating vertices

    Returns:
        [type] -- returns new mesh (vertices, edges) with corresponding trace map
    """
    bins = coords // voxel_size
    unique_bins, reverse_trace_ids = np.unique(
        bins, axis=0, return_inverse=True)

    edge_map = defaultdict(set)
    edge_output = []
    traces = []

    for bin_id in range(len(unique_bins)):
        points_in_bin = np.argwhere(reverse_trace_ids == bin_id).flatten()

        traces.append(points_in_bin)

        for point_id in points_in_bin:
            try:
                neigh = edges[point_id]
            except IndexError:
                print('WARNING: single point with no connections')
                continue
            new_edges = reverse_trace_ids[neigh]
            edge_map[bin_id].update(set(new_edges))

    new_edge_list = []

    for key in sorted(edge_map):
        edge_map[key].discard(key)
        value = list(edge_map[key])
        new_edge_list.append(value)

        for elem in value:
            edge_output.append([key, elem])

    new_coords = np.empty((len(traces), 3), dtype=np.float32)

    for index, trace in enumerate(traces):
        # take center of gravity for each cell for the new representative vertex
        new_coords[index, :] = coords[trace].mean(axis=0)

    return new_coords, reverse_trace_ids, new_edge_list, edge_output


def quadric_error_metric(curr_file_path, ratio, old_vertices):
    os.system(f"tridecimator "
              f"{curr_file_path} {curr_file_path} {ratio} -On -C > /dev/null")

    mesh = open3d.read_triangle_mesh(curr_file_path)
    if not mesh.has_vertices():
        raise QEMError('no vertices left')
    coords = np.asarray(mesh.vertices)
    edges_list = edges_from_faces(np.asarray(mesh.triangles))
    edge_out = []
    for key, group in enumerate(edges_list):
        for elem in group:
            edge_out.append([key, elem])

    reverse_trace = csv2npy(curr_file_path.replace(
        '.ply', '.csv'), old_vertices=old_vertices, new_vertices=coords)

    return coords, edge_out, reverse_trace


def nearest_neighbor_interpolation_for_unassigned_traces(new_coords, old_coords, trace_scatter):
    try:
        points_in_question = old_coords[trace_scatter == -1]
        if points_in_question.shape[0] == 0:
            return trace_scatter

        ball_tree = BallTree(new_coords[:, :3])
        trace = ball_tree.query(points_in_question, k=1)[1].flatten()
        trace_scatter[trace_scatter == -1] = trace
        return trace_scatter
    except:
        raise QEMError('error in nn interpolation')


def process_frame(file_path: str, global_params: dict = None):
    premapping = global_params['mapping']

    original_mesh = open3d.read_triangle_mesh(file_path)
    original_mesh.compute_vertex_normals()

    if args.train:
        # create train/val dataset
        if args.dataset in ['scannet']:
            labels_file_path = file_path.replace('.ply', '.labels.ply')
            vertex_labels = np.asarray(PlyData.read(
                labels_file_path)['vertex']['label'])
        elif args.dataset in ['s3dis']:
            labels_file_path = file_path.replace('.ply', '.labels.npy')
            vertex_labels = np.load(labels_file_path)
        elif args.dataset in ['matterport']:
            mapped_labels = premapping[PlyData.read(
                file_path)['face']['category_id']]
            mapped_labels[np.logical_not(
                np.isin(mapped_labels, MATTERPORT_ALLOWED_NYU_CLASSES))] = 0

            vertices = np.asarray(original_mesh.vertices)
            triangles = np.asarray(original_mesh.triangles)

            remapped_labels = MATTERPORT_CLASS_REMAP[mapped_labels].astype(int)

            vertex_labels = np.zeros((vertices.shape[0], 22), dtype=np.int)

            # in matterport, the labels are given for each face and not for each vertex
            for row_id in range(triangles.shape[0]):
                for i in range(3):
                    vertex_labels[triangles[row_id][i],
                                  remapped_labels[row_id]] += 1

            vertex_labels = np.argmax(vertex_labels, axis=1)

        original_vertices = np.column_stack(
            (np.asarray(original_mesh.vertices),
             np.asarray(original_mesh.vertex_colors),
             np.asarray(original_mesh.vertex_normals),
             vertex_labels))

        if args.dataset in ['scannet']:
            # FIX: THREE MESHES HAVE CORRUPTED LABEL IDS
            class_ids = original_vertices[:, -1].astype(int)
            class_ids[class_ids > 40] = 0
            original_vertices[:, -1] = SCANNET_CLASS_REMAP[class_ids]
    else:
        # create test dataset (labels are not given here!)
        # this can only be done with scannet
        assert args.dataset in ['scannet']
        original_vertices = np.column_stack((np.asarray(original_mesh.vertices),
                                             np.asarray(
                                                 original_mesh.vertex_colors),
                                             np.asarray(original_mesh.vertex_normals)))

        # store the number of vertices in order to generate dummy placeholders while evaluating on the test set
        s_path = file_path.replace('.ply', '_size.txt')
        if not os.path.isfile(s_path):
            with open(s_path, 'w') as s_file:
                s_file.write(f"{original_vertices.shape[0]}")

    if args.dataset in ['scannet']:
        subfolder = f"{file_path.split('/')[-2]}"
        curr_dir = f"{args.out_path}{subfolder}"
    elif args.dataset in ['matterport']:
        subfolder = f"{file_path.split('/')[-3]}_{file_path.split('/')[-1].replace('.ply', '')}"
        curr_dir = f"{args.out_path}{subfolder}"
    elif args.dataset in ['s3dis']:
        subfolder = f"{file_path.split('/')[-3]}_{file_path.split('/')[-2]}"
        curr_dir = f"{args.out_path}{subfolder}"

    clear_folder(f"{curr_dir}/")

    coords = []
    edges_list = []
    edge_output = []
    traces = []

    curr_mesh = original_mesh

    curr_vertices = np.asarray(curr_mesh.vertices)

    edge_list_0 = edges_from_faces(np.asarray(curr_mesh.triangles))

    coords.append(curr_vertices)
    edges_list.append(edge_list_0)

    edge_output_0 = []
    for key, group in enumerate(edge_list_0):
        for elem in group:
            edge_output_0.append([key, elem])

    edge_output.append(np.array(edge_output_0))

    if not args.vertex_clustering:
        # put current mesh in the working directory
        curr_mesh_path = f"{curr_dir}/curr_mesh.ply"
        open3d.io.write_triangle_mesh(curr_mesh_path, curr_mesh)

    for level in range(len(args.level_params)):
        if args.vertex_clustering:
            coords_l, trace_scatter, edge_list_l, edge_output_l = \
                vertex_clustering(
                    coords[-1], edges_list[-1], float(args.level_params[level]))
        else:
            if not args.level_params[level].isdigit():
                # before using QEM apply Vertex Clustering with ensures triangular mesh
                # TODO no absolute paths!
                os.system(f"trimesh_clustering "
                          f"{curr_mesh_path} {curr_mesh_path} -s {args.level_params[level]} > /dev/null")
                curr_mesh = open3d.io.read_triangle_mesh(curr_mesh_path)
                curr_mesh.compute_vertex_normals()

                coords_l = np.asarray(curr_mesh.vertices)

                edge_list_l = edges_from_faces(np.asarray(curr_mesh.triangles))

                edge_output_0 = []
                for key, group in enumerate(edge_list_l):
                    for elem in group:
                        edge_output_0.append([key, elem])

                edge_output_l = edge_output_0

                vh_ball_tree = BallTree(coords_l[:, :3])
                vh_trace = vh_ball_tree.query(
                    np.asarray(coords[0][:, :3]), k=1)[1].flatten()
                trace_scatter = vh_trace
            else:
                coords_l, edge_output_l, trace_scatter = \
                    quadric_error_metric(curr_mesh_path, int(args.level_params[level]),
                                         old_vertices=coords[-1])
                edge_list_l = None

        coords.append(coords_l)
        traces.append(trace_scatter)
        edges_list.append(edge_list_l)
        edge_output.append(np.array(edge_output_l))

    colors_labels = get_color_and_labels(original_vertices, coords)

    coords_color_labels = []

    for i in range(len(coords)):
        coords_color_labels.append(
            np.column_stack((coords[i], colors_labels[i])))

    clear_folder(f"{curr_dir}/")

    coords_color_labels = [torch.from_numpy(
        coords_color_labels[i]) for i in range(len(coords_color_labels))]

    pt_data = {}

    if args.train:
        vertices = [coords_color_labels[1][:, :-1].float()]
        vertices.extend([coords_color_labels[i][:, :3].float()
                         for i in range(2, len(coords_color_labels))])
        labels = coords_color_labels[0][:, -1].long()

        pt_data['vertices'] = vertices
        pt_data['labels'] = labels
    else:
        vertices = [coords_color_labels[1].float()]
        vertices.extend([coords_color_labels[i][:, :3].float()
                         for i in range(2, len(coords_color_labels))])
        pt_data['vertices'] = vertices

    pt_data['edges'] = [torch.from_numpy(
        edge_output[i]).long() for i in range(1, len(edge_output))]
    pt_data['traces'] = [torch.from_numpy(x).long() for x in traces]

    torch.save(pt_data, f"{curr_dir}.pt")

    # DELETE EMPTY FOLDER
    shutil.rmtree(f"{curr_dir}/")


def main(args):
    assert args.dataset in ['scannet', 's3dis', 'matterport']

    mapping = dict()

    if args.dataset in ['scannet']:
        if args.train:
            if not args.val:
                considered_rooms_path = '../../dataset/meta/scannet/scannetv2_train.txt'
                with open(considered_rooms_path, 'r') as f:
                    considered_rooms = f.read().splitlines()
            else:
                considered_rooms_path = '../../dataset/meta/scannet/scannetv2_val.txt'
                with open(considered_rooms_path, 'r') as f:
                    considered_rooms = f.read().splitlines()
        else:
            considered_rooms_path = '../../dataset/meta/scannet/scannetv2_test.txt'
            with open(considered_rooms_path, 'r') as f:
                considered_rooms = f.read().splitlines()

    if args.dataset in ['s3dis']:
        file_paths = sorted([x for
                             x in glob.glob(f"{args.in_path}/*/*/*.ply")])
    elif args.dataset in ['scannet']:
        file_paths = sorted([x for
                             x in glob.glob(f"{args.in_path}/*/*.ply")
                             if 'clean_2.ply' in x
                             and x.split('/')[-1].rsplit('_', 3)[0] in considered_rooms])
    elif args.dataset == 'matterport':
        category_mapping = pd.read_csv('../../dataset/meta/matterport/category_mapping.tsv',
                                       sep='\t', header=0)

        mapping = np.insert(category_mapping[['nyu40id']].to_numpy()
                            .astype(int).flatten(), 0, 0, axis=0)

        file_paths = sorted([x for
                             x in glob.glob(f"{args.in_path}/*/region_segmentations/*.ply")])

    process_frame(file_paths[args.number], global_params={'mapping': mapping})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description="Create hierarchy of graph levels with a specified mesh simplification algorithm")

    parser.add_argument('--in_path', type=str, required=True,
                        help='path to the root directory of folders containing the original dataset')

    parser.add_argument('--out_path', type=str, required=True,
                        help='path to the directory where the simplified meshes and tracing files should be stored')

    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.set_defaults(train=True)

    parser.add_argument('--val', dest='val', action='store_true')
    parser.set_defaults(val=False)

    parser.add_argument('--vertex_clustering',
                        dest='vertex_clustering', action='store_true')
    parser.add_argument('--qem', dest='vertex_clustering',
                        action='store_false')
    parser.set_defaults(vertex_clustering=True)

    parser.add_argument('--level_params', required=True, nargs='+',
                        help='voxel sizes per graph level (VC) / ratio of kept vertices (QEM)')

    parser.add_argument('--dataset', type=str, required=True,
                        help='name of the dataset, e.g. [scannet, matterport, s3dis]')

    parser.add_argument('--number', const=-1, default=-1, type=int, nargs='?',
                        help='number of task id (used for parallization with SLURM)')

    args = parser.parse_args()
    pretty_print_arguments(args)
    open3d.set_verbosity_level(open3d.VerbosityLevel.Error)
    main(args)
