import torch
from torch.nn import Sequential as Seq, Linear as Lin, functional as F, ReLU, LeakyReLU, BatchNorm1d
from torch_scatter import scatter_mean, scatter_max
from base.base_model import BaseModel
from model.module.edge_conv_translation_invariance import EdgeConvTransInv
from model.module.get_gcn_filter import get_gcn_filter


class DualConvMeshNet(BaseModel):
    """DualConvMeshNet: Joint geodesic and Euclidean convolutions in a U-Net like architecture on 3D meshes"""

    def __init__(self, feature_number, num_propagation_steps, geo_filter, euc_filter,
                 activation, num_classes, pooling_method='mean', aggr='mean', one_neighborhood=False):
        super(DualConvMeshNet, self).__init__()

        assert len(geo_filter) == len(euc_filter)

        curr_size = feature_number

        inplace = False
        self._pooling_method = pooling_method
        self._one_neighborhood = one_neighborhood

        if activation == 'ReLU':
            self._activation = ReLU
            self._act = F.relu
        elif activation == 'LeakyReLU':
            self._activation = LeakyReLU
            self._act = F.leaky_relu
        else:
            raise NotImplementedError(f"{activation} is not implemented")

        self.left_geo_cnns = []
        self.left_euc_cnns = []

        self.right_geo_cnns = []
        self.right_euc_cnns = []

        self._graph_levels = len(geo_filter)

        for level in range(len(geo_filter)):
            if level < len(geo_filter) - 1:
                if level == 0:
                    # First level needs translation invariant version of edge conv
                    left_geo = [get_gcn_filter(
                        curr_size, geo_filter[level], self._activation, aggregation=aggr, module=EdgeConvTransInv)]
                    left_euc = [get_gcn_filter(
                        curr_size, euc_filter[level], self._activation, aggregation=aggr, module=EdgeConvTransInv)]
                else:
                    left_geo = [get_gcn_filter(
                        2 * curr_size, geo_filter[level], self._activation, aggregation=aggr)]
                    left_euc = [get_gcn_filter(
                        2 * curr_size, euc_filter[level], self._activation, aggregation=aggr)]

                for _ in range(num_propagation_steps - 1):
                    left_geo.append(get_gcn_filter(2*(geo_filter[level] + euc_filter[level]), geo_filter[level],
                                                   self._activation, aggregation=aggr))
                    left_euc.append(get_gcn_filter(2*(geo_filter[level] + euc_filter[level]), euc_filter[level],
                                                   self._activation, aggregation=aggr))

                # DECODER branch of U-NET
                curr_size = geo_filter[level] + euc_filter[level] + \
                    geo_filter[level+1] + euc_filter[level+1]

                right_geo = [get_gcn_filter(
                    2 * curr_size, geo_filter[level], self._activation, aggregation=aggr)]
                right_euc = [get_gcn_filter(
                    2 * curr_size, euc_filter[level], self._activation, aggregation=aggr)]

                for _ in range(num_propagation_steps - 1):
                    right_geo.append(get_gcn_filter(2*(geo_filter[level] + euc_filter[level]), geo_filter[level],
                                                    self._activation, aggregation=aggr))
                    right_euc.append(get_gcn_filter(2*(geo_filter[level] + euc_filter[level]), euc_filter[level],
                                                    self._activation, aggregation=aggr))

                self.right_geo_cnns.append(torch.nn.ModuleList(right_geo))
                self.right_euc_cnns.append(torch.nn.ModuleList(right_euc))

                curr_size = geo_filter[level] + euc_filter[level]
            else:
                left_geo = [get_gcn_filter(
                    2 * curr_size, geo_filter[level], self._activation, aggregation=aggr)]
                left_euc = [get_gcn_filter(
                    2 * curr_size, euc_filter[level], self._activation, aggregation=aggr)]

                for _ in range(num_propagation_steps - 1):
                    left_geo.append(get_gcn_filter(2 * (geo_filter[level] + euc_filter[level]), geo_filter[level],
                                                   self._activation, aggregation=aggr))
                    left_euc.append(get_gcn_filter(2 * (geo_filter[level] + euc_filter[level]), euc_filter[level],
                                                   self._activation, aggregation=aggr))

            self.left_geo_cnns.append(torch.nn.ModuleList(left_geo))
            self.left_euc_cnns.append(torch.nn.ModuleList(left_euc))

        self.final_convs = [
            Seq(
                Lin(geo_filter[0] + euc_filter[0],
                    (geo_filter[0] + euc_filter[0]) // 2),
                BatchNorm1d((geo_filter[0] + euc_filter[0]) // 2),
                self._activation(inplace=inplace),
                Lin((geo_filter[0] + euc_filter[0]) // 2, num_classes)
            )
        ]

        self.left_geo_cnns = torch.nn.ModuleList(self.left_geo_cnns)
        self.left_euc_cnns = torch.nn.ModuleList(self.left_euc_cnns)

        self.right_geo_cnns = torch.nn.ModuleList(self.right_geo_cnns)
        self.right_euc_cnns = torch.nn.ModuleList(self.right_euc_cnns)

        self.final_convs = torch.nn.ModuleList(self.final_convs)

    def _residual_steps(self, vertex_features, geo_filters, eucl_filters, geo_edges, eucl_edges, inplace=False):
        residual_geo = geo_filters[0](vertex_features, geo_edges)
        residual_eucl = eucl_filters[0](vertex_features, eucl_edges)
        vertex_features = self._act(
            torch.cat((residual_geo, residual_eucl), dim=-1), inplace=inplace)

        for step in range(1, len(geo_filters)):
            residual_geo = geo_filters[step](vertex_features, geo_edges)
            residual_eucl = eucl_filters[step](vertex_features, eucl_edges)
            residual_cat = torch.cat((residual_geo, residual_eucl), dim=-1)
            vertex_features += residual_cat
            vertex_features = self._act(vertex_features, inplace=inplace)
        return vertex_features

    def _simple_residual_steps(self, vertex_features, filters, edges, inplace=False):
        residual = filters[0](vertex_features, edges)
        vertex_features = self._act(residual, inplace=inplace)

        for step in range(1, len(filters)):
            residual = filters[step](vertex_features, edges)
            vertex_features += residual
            vertex_features = self._act(vertex_features, inplace=inplace)
        return vertex_features

    def _pooling(self, vertex_features, edges):
        if self._pooling_method == 'mean':
            return scatter_mean(vertex_features, edges, dim=0)
        if self._pooling_method == 'max':
            return scatter_max(vertex_features, edges, dim=0)[0]

        raise ValueError(f"Unkown pooling type {self._pooling_method}")

    def forward(self, sample):
        # if network operates in only one domain, assume all neighborhood information is stored in edge_index
        if self._one_neighborhood:
            sample[f"euclidean_edge_index"] = sample.edge_index

            for level in range(1, self._graph_levels):
                sample[f"hierarchy_euclidean_edge_index_{level}"] = sample[f"hierarchy_edge_index_{level}"]

        levels = []
        level1 = torch.cat((sample.pos, sample.x), dim=-1)
        level1 = self._residual_steps(level1, self.left_geo_cnns[0], self.left_euc_cnns[0],
                                      sample.edge_index, sample.euclidean_edge_index)

        levels.append(level1)

        # ENCODER BRANCH
        for level in range(1, self._graph_levels):
            curr_level = self._pooling(levels[-1],
                                       sample[f"hierarchy_trace_index_{level}"])

            curr_level = self._residual_steps(curr_level, self.left_geo_cnns[level], self.left_euc_cnns[level],
                                              sample[f"hierarchy_edge_index_{level}"],
                                              sample[f"hierarchy_euclidean_edge_index_{level}"])

            levels.append(curr_level)

        current = levels[-1]

        # DECODER BRANCH
        for level in range(1, self._graph_levels):
            back = current[sample[f"hierarchy_trace_index_{self._graph_levels - level}"]]
            fused = torch.cat((levels[-(level+1)], back), -1)

            if level == self._graph_levels - 1:
                fused = self._residual_steps(fused, self.right_geo_cnns[-level], self.right_euc_cnns[-level],
                                             sample.edge_index, sample.euclidean_edge_index)
            else:
                fused = self._residual_steps(fused, self.right_geo_cnns[-level], self.right_euc_cnns[-level],
                                             sample[f"hierarchy_edge_index_{self._graph_levels - level - 1}"],
                                             sample[f"hierarchy_euclidean_edge_index_{self._graph_levels - level - 1}"])
            current = fused

        result = current

        for conv in self.final_convs:
            result = conv(result)

        return result
