from torch_geometric.utils import remove_self_loops, add_self_loops


class AddSelfLoops:
    """Some convolutions need self loop information. Add them to the edge set."""

    def __init__(self, identifier: str = 'edge_index'):
        """Initialize self loop adder

        Keyword Arguments:
            identifier {str} -- identify set to which self loops should be added (default: {'edge_index'})
        """
        self.identifier = identifier

    def __call__(self, sample):
        keys = [x for x in dir(sample) if self.identifier in x]

        for key in keys:
            if str.isdigit(key[-1]):
                pos_id = f"pos_{key[-1]}"
            else:
                pos_id = 'pos'

            sample[key], _ = remove_self_loops(sample[key])
            sample[key], _ = add_self_loops(
                sample[key], num_nodes=sample[pos_id].size(0))

        return sample
