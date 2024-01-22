# TODO: recheck imports

import torch
from utils.utils import remove_edges
from utils.graph_viewer import plot_weights
from gat_model import gat_model
import torch_geometric.utils as utils
from torch_geometric.nn import GAE


def train_network_gae(gae, optimizer, graph):
    gae.train()
    optimizer.zero_grad()

    att_tuple, H_L = gae.encode(graph.features.float(), graph.edge_index)

    # Decode por multiplicação pela transposta
    loss = gae.recon_loss(H_L, graph.edge_index)

    return float(loss), H_L, att_tuple


def run_training(epochs, G, communities, dataset_to_use):
    # TODO: move all this training code to the right method
    device = torch.device('cpu')

    dataset = utils.from_networkx(G)

    in_channels, hidden_channels, out_channels = len(dataset.features[0]), 8, 2

    gae = GAE(gat_model.GATLayer(in_channels, hidden_channels, out_channels))

    gae = gae.to(device)
    gae = gae.float()

    dataset = dataset.to(device)

    optimizer = torch.optim.Adam(gae.parameters(), lr=0.005)

    losses = []
    embs_list = []

    for epoch in range(epochs):
        loss, H_L, att_tuple = train_network_gae(gae, optimizer, dataset)
        if epoch % 10 == 0:
            print("Loss:", loss)
        losses.append(loss)
        embs_list.append(H_L)

    # Add the Attention values to the original Graph edges
    weight = att_tuple[1]
    src = att_tuple[0][0]
    tgt = att_tuple[0][1]

    for i in range(len(weight)):
        G.add_edge(src[i].item(), tgt[i].item(), weight=weight[i].item())

    # Plot original graph with edge weights
    # plot_weights(G, communities)

    if dataset_to_use == "cora":
        G, communities = remove_edges(G, communities, num_edges_to_remove=6000)
    else:
        G, communities = remove_edges(G, communities)

    plot_weights(G, communities)
