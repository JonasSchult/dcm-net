"""First convolution of the network needs to be translation invariant.
Therefore, we do not concatenate the initial features of the vertices which include
absolute 3D positions.
"""
import torch_geometric.nn.conv.edge_conv as edge
import torch


class EdgeConvTransInv(edge.EdgeConv):
    """First convolution of the network needs to be translation invariant.
    Therefore, we do not concatenate the initial features of the vertices which include
    absolute 3D positions.
    """

    def __init__(self, nn, aggr):
        super(EdgeConvTransInv, self).__init__(nn, aggr)
        self._aggr = aggr

    def message(self, x_i, x_j):
        # do not concatenate x_i features; in the first conv, these are absolute position values!
        return self.nn(torch.cat([x_j - x_i], dim=1))

    def __repr__(self):
        return '{}(nn={}, aggr={})'.format(self.__class__.__name__, self.nn, self._aggr)
