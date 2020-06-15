from torch_geometric.nn import radius_graph
from typing import List


class RadiusNeighbors:
    """Returns radius neighbors for all vertices in a sample."""

    def __init__(self, radius: List[float], max_neigh: int = 32, override: bool = False,
                 keep_pos: bool = False, visualization: bool = False):
        """Initialize radius neighborhoods for all vertices in all hierarchy levels

        Arguments:
            radius {List[float]} -- list of radiuses for each hierarchy level

        Keyword Arguments:
            max_neigh {int} -- maximal number of radius neighbors.
                               Important to keep preprocessing runtime and memory footprint low (default: {32})
            override {bool} -- override edge set -> for SingleConv architectures (default: {False})
            keep_pos {bool} -- keep positions of vertices in later hierarchy levels (default: {False})
            visualization {bool} -- only calculate radius neighbors in first hierarchy level for visualization (default: {False})
        """
        self._radius = radius
        self._override = override
        self._max_neigh = max_neigh
        self._keep_pos = keep_pos
        self._visualization = visualization

    def __call__(self, sample):
        if self._visualization:
            keys = sorted([x for x in dir(sample) if 'edge_index' == x])
        else:
            keys = sorted(
                [x for x in dir(sample) if 'edge_index' in x and 'dilated' not in x])

        pos_keys = sorted([x for x in dir(sample) if 'pos' in x])

        for level, key in enumerate(keys):
            radius_edges = radius_graph(
                sample[pos_keys[level]], self._radius[level], max_num_neighbors=self._max_neigh)

            if not self._override:
                sample[key.replace(
                    'edge_index', 'euclidean_edge_index')] = radius_edges
            else:
                # in a single conv architecture, we only have to keep track of one edge set
                sample[key] = radius_edges

        if not self._keep_pos:
            # we do not need to now actual spatial positions of later layers -> delete them
            keys = sorted([x for x in dir(sample) if 'pos_' in x])
            for key in keys:
                delattr(sample, key)

        return sample
