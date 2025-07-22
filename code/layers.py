import gol
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import softmax
from torchsde import sdeint

class GeoConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GeoConv, self).__init__(aggr='add')
        self._cached_edge = None
        self.lin = nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, geo_graph: Data):
        if self._cached_edge is None:
            self._cached_edge = gcn_norm(geo_graph.edge_index, add_self_loops=False)
        edge_index, norm_weight = self._cached_edge
        # return x
        x = self.lin(x)

        return self.propagate(edge_index, x=x, norm=norm_weight, dist_vec=geo_graph.edge_attr)

    def message(self, x_j, norm, dist_vec):
        return norm.unsqueeze(-1) * x_j * dist_vec.unsqueeze(-1)


class SeqConv(MessagePassing):
    def __init__(self, hid_dim, flow="source_to_target"):
        super(SeqConv, self).__init__(aggr='add', flow=flow)
        self.hid_dim = hid_dim
        self.alpha_src = nn.Linear(hid_dim, 1, bias=False)
        self.alpha_dst = nn.Linear(hid_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.alpha_src.weight)
        nn.init.xavier_uniform_(self.alpha_dst.weight)

        self.act = nn.LeakyReLU()

    def forward(self, embs, seq_graph):
        # node_embs, distance_embs, temporal_embs = embs
        node_embs, distance_embs = embs
        sess_idx, edge_index, batch_idx = seq_graph.x.squeeze(), seq_graph.edge_index, seq_graph.batch
        edge_dist =  seq_graph.edge_dist

        x = node_embs[sess_idx]
        # return x
        edge_l = distance_embs[edge_dist]

        all_edges = torch.cat((edge_index, edge_index[[1, 0]]), dim=-1)
        seq_embs = self.propagate(all_edges, x=x, edge_l=edge_l, edge_size=edge_index.size(1))
        return seq_embs

    def message(self, x_j, x_i, edge_index_i, edge_l, edge_size):
        element_sim = x_j * x_i
        src_logits = self.alpha_src(element_sim[: edge_size] + edge_l ).squeeze(-1)
        tot_logits = torch.cat((src_logits, src_logits))
        attn_weight = softmax(tot_logits, edge_index_i)
        aggr_embs = x_j * attn_weight.unsqueeze(-1)
        return aggr_embs


def sequence_mask(lengths, max_len=None) -> torch.Tensor:
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)
