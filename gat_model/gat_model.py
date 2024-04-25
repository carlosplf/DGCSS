import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F


# Creating a GAT Network.
class GATLayer(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATLayer, self).__init__()

        self.gat_conv1 = GATConv(in_channels, hidden_channels,
                                 heads=heads, add_self_loops=True)
        self.gat_conv2 = GATConv(hidden_channels * heads, out_channels,
                                 heads=1, add_self_loops=True)

    # IMPORTANT: Just the Att matrix from one layer is being returned by this method.
    def forward(self, x, edge_index, weights=None):
        x, att_tuple = self.gat_conv1(x, edge_index, weights, return_attention_weights=True)
        x = F.relu(x)
        x = self.gat_conv2(x, edge_index, weights)

        return att_tuple, x
