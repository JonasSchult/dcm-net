from torch_geometric.utils import degree
import torch
from torch_sparse import coalesce
import math

#TODO CHECK INFLUENCE OF THIS


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index,
                edge_attr=None,
                force_undirected=False,
                num_nodes=None,
                degrees=None,
                cutoff=10,
                alpha=1.):

    N = num_nodes
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    filter = (degrees > cutoff)[row].float()
    keep_probability = filter * torch.pow((degrees[row] + 1 - cutoff).float(), - alpha / math.log(cutoff+1, 2))

    keep_probability[(1-filter).byte()] = 1.

    mask = torch.bernoulli(keep_probability).byte()

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)],
            dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


class EdgeSampling:
    def __init__(self, cutoff, alpha):
        self._cutoff = cutoff
        self._alpha = alpha

    def __call__(self, sample):
        edge_idx = [a for a in dir(sample) if 'edge_index' in a]

        for edge_id in edge_idx:
            num_nodes = sample[edge_id].max().item() + 1
            row, _ = sample[edge_id]
            node_degrees = degree(row, num_nodes, dtype=sample[edge_id].dtype)

            new_edges, _ = dropout_adj(sample[edge_id],
                                       force_undirected=False,
                                       num_nodes=num_nodes,
                                       degrees=node_degrees,
                                       cutoff=self._cutoff,
                                       alpha=self._alpha)

            sample[edge_id] = new_edges

        return sample
