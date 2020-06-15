from torch_geometric.transforms import TwoHop
from torch_geometric.data import Data
import torch
from torch_geometric.nn import knn_graph, knn
from torch_scatter import scatter_mean
from torch_cluster.fps import *


class FPSPRadius:
    def __init__(self):
        pass

    def __call__(self, sample):
        keys = sorted([x for x in dir(sample) if 'edge_index' in x])

        num_vertices = [sample.num_nodes] + sample.num_vertices

        # sample.edge_index = knn_graph(sample.pos, self._k[0])

        # knn_edges = knn_graph(sample.pos, self._k[0] * self._d)
        # dilated_idx = [index for index in range(knn_edges.shape[1])[0::self._d]]

        # sample.edge_index = knn_edges[:, dilated_idx]

        for level, key in enumerate(keys):
            if level == len(keys) - 1:
                break

            pos_key = key.replace('edge_index', 'pos').replace('hierarchy_', '')

            subset_points_idx = fps(sample[pos_key], ratio=sample.num_vertices[level] / num_vertices[level])

            # if level == 0:
            #     sample.y = sample.y[subset_points_idx]

            num_vertices[level+1] = subset_points_idx.shape[0]
            sample.num_vertices[level] = num_vertices[level+1]
            sample['pos_' + str(level+1)] = sample[pos_key][subset_points_idx]
            sample[f"hierarchy_trace_index_{level+1}"] = knn(sample['pos_' + str(level+1)], sample[pos_key], 1)[1, :]
            # sample[f"hierarchy_edge_index_{level+1}"] = knn_graph(sample['pos_' + str(level+1)], self._k[level+1])

            # knn_edges = knn_graph(sample['pos_' + str(level+1)], self._k[level+1] * self._d)
            # dilated_idx = [index for index in range(knn_edges.shape[1])[0::self._d]]

            # sample[f"hierarchy_edge_index_{level+1}"] = knn_edges[:, dilated_idx]

        keys = sorted([x for x in dir(sample) if x.startswith('x_')])
        for key in keys:
            delattr(sample, key)

        # keys = sorted([x for x in dir(sample) if 'pos_' in x])
        # for key in keys:
        #     delattr(sample, key)

        return sample
