import torch
import logging
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GATLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATLayer, self).__init__()

        self.gat_conv1 = GATConv(in_channels, hidden_channels, add_self_loops=True)
        self.gat_conv2 = GATConv(hidden_channels, out_channels, add_self_loops=True)

        msg = (
            "Network structure: "
            + str(in_channels)
            + ", "
            + str(hidden_channels)
            + ", "
            + str(out_channels)
        )
        logging.info(msg)

    # IMPORTANT: Just the Att matrix from one layer is being returned from this method.
    def forward(self, x, edge_index, weights=None):
        x, att_tuple = self.gat_conv1(
            x, edge_index, weights, return_attention_weights=True
        )
        x = F.relu(x)
        x = self.gat_conv2(x, edge_index, weights)

        return att_tuple, x


class GAT2Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT2Layer, self).__init__()

        self.gat_conv1 = GATConv(in_channels, hidden_channels[0], add_self_loops=True)
        self.gat_conv2 = GATConv(
            hidden_channels[0], hidden_channels[1], add_self_loops=True
        )
        self.gat_conv3 = GATConv(hidden_channels[1], out_channels, add_self_loops=True)

        msg = (
            "Network structure: "
            + str(in_channels)
            + ", "
            + str(hidden_channels)
            + ", "
            + str(out_channels)
        )
        logging.info(msg)

    # IMPORTANT: Just the Att matrix from one layer is being returned from this method.
    def forward(self, x, edge_index, weights=None):
        x, att_tuple = self.gat_conv1(
            x, edge_index, weights, return_attention_weights=True
        )
        x = F.relu(x)
        x = self.gat_conv2(x, edge_index, weights)
        x = F.relu(x)
        x = self.gat_conv3(x, edge_index, weights)

        return att_tuple, x
