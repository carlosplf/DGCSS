# TODO: recheck imports

import torch
from torch_geometric.nn import GAE
from torch_geometric.utils import to_networkx

from gat_model import gat_model
from utils.graph_viewer import plot_weights
from utils.utils import remove_edges


def train_network_gae(gae, optimizer, data):
    gae.train()
    optimizer.zero_grad()

    att_tuple, H_L = gae.encode(data.features.float(), data.edge_index)

    # Decode por multiplicação pela transposta
    loss = gae.recon_loss(H_L, data.edge_index)

    loss.backward()
    optimizer.step()

    return float(loss), H_L, att_tuple


def run_training(epochs, data, features, exp_id=0):
    # TODO: move all this training code to the right method
    device = torch.device("cpu")

    in_channels, hidden_channels, out_channels = 5, 32, 8

    gae = GAE(gat_model.GATLayer(in_channels, hidden_channels, out_channels))

    gae = gae.to(device)
    gae = gae.float()

    data = data.to(device)

    optimizer = torch.optim.Adam(gae.parameters(), lr=0.001)

    losses = []
    embs_list = []
    att_tuple = [[]]

    for epoch in range(epochs):
        loss, H_L, att_tuple = train_network_gae(gae, optimizer, data)
        if epoch % 10 == 0:
            print("==>", epoch, "- Loss:", loss)
        losses.append(loss)
        embs_list.append(H_L)

    # plt.plot(losses, label="train_loss")
    # plt.show()

    # Add the Attention values to the original Graph edges
    weight = att_tuple[1]
    src = att_tuple[0][0]
    tgt = att_tuple[0][1]

    # After running all the training, transform the dataset to Graph
    # TODO: Do we need to transform as a Graph again? It was a Graph before
    # we transform to Dataset for training.
    G = to_networkx(data, to_undirected=True)

    # Define the edges weights as the values from the Attention matrix
    for i in range(len(weight)):
        G.add_edge(src[i].item(), tgt[i].item(), weight=weight[i].item())

    # Plot original graph with edge weights
    filename = "before-" + str(exp_id) + ".png"
    plot_weights(G, features, folder_path="./charts", filename=filename)

    group_size_fraction = 5
    G = remove_edges(G, group_size_fraction)

    filename = "after-" + str(exp_id) + ".png"
    plot_weights(G, features, folder_path="./charts", filename=filename)
