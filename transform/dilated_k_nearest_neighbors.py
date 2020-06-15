from torch_geometric.nn import knn_graph


class DilatedKNearestNeighbors:
    """Returns dilated k-nearest neighbors. That is, every d-th neighbor."""

    def __init__(self, k=10, d: int = 8, override:bool = False, no_pos:bool = True, visualization:bool = False):
        """Initialize dilated knn.
        
        Keyword Arguments:
            k {int} -- Return k neighbors (default: {10})
            d {int} -- dilation factor (every d-th neighbor) (default: {8})
            override {bool} -- override edge set -> for SingleConv architectures (default: {False})
            keep_pos {bool} -- keep positions of vertices in later hierarchy levels (default: {False})
            visualization {bool} -- only calculate radius neighbors in first hierarchy level for visualization (default: {False})
        """

        # if integer given then use it for all hierarchy levels
        try:
            self._k = [int(k)]
        except TypeError:
            self._k = k

        self._override = override
        self._d = d
        self._no_pos = no_pos
        self._visualization = visualization

    def __call__(self, sample):
        if self._visualization:
            keys = sorted([x for x in dir(sample) if 'edge_index' == x])
        else:
            keys = sorted([x for x in dir(sample) if 'edge_index' in x and 'dilated' not in x])

        pos_keys = sorted([x for x in dir(sample) if 'pos' in x])

        if len(self._k) == 1:
            # assume the same 'k' for all hierarchy levels
            self._k = [self._k[0] for _ in range(len(pos_keys))]

        for level, key in enumerate(keys):
            knn_edges = knn_graph(sample[pos_keys[level]], k=self._k[level] * self._d)
            dilated_idx = [index for index in range(knn_edges.shape[1])[0::self._d]]

            if not self._override:
                sample[key.replace('edge_index', 'euclidean_edge_index')] = knn_edges[:, dilated_idx]
            else:
                sample[key] = knn_edges[:, dilated_idx]

        if self._no_pos:
            keys = sorted([x for x in dir(sample) if 'pos_' in x])
            for key in keys:
                delattr(sample, key)

        return sample
