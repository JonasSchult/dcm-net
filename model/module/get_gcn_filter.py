"""Function definitions for graph conv modules used in our DCM Net
"""
import torch
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.nn.conv.edge_conv as edge

def get_gcn_filter(input_size: int, output_size, activation: torch.nn.Module,
                   inplace: bool = False, aggregation: str = 'mean',
                   module: MessagePassing = edge.EdgeConv):
    """returns graph conv module with specified arguments and type

    Arguments:
        input_size {int} -- input size (2 * current vertex feature size!)
        output_size {[type]} -- feature size of new vertex features
        activation {torch.nn.Module} -- activation function for internal MLP

    Keyword Arguments:
        inplace {bool} -- (default: {False})
        aggregation {str} -- permutation-invariant feature aggregation of adjacent vertices (default: {'mean'})
        module {MessagePassing} -- graph convolutional module (default: {edge.EdgeConv})
    """

    assert input_size >= 0
    assert output_size >= 0

    inner_module = Seq(
        Lin(input_size, 2 * output_size),
        BatchNorm1d(2 * output_size),
        activation(inplace=inplace),
        Lin(2 * output_size, output_size),
        BatchNorm1d(output_size))

    return module(inner_module, aggr=aggregation)
