# TODO: recheck imports

import torch
from torch_geometric.nn import GAE
from gat_model import gat_model


def train_network_gae(gae, optimizer, data):
    gae.train()
    optimizer.zero_grad()

    att_tuple, H_L = gae.encode(data.x.float(), data.edge_index)

    # Decode por multiplicação pela transposta
    loss = gae.recon_loss(H_L, data.edge_index)

    loss.backward()
    optimizer.step()

    return float(loss), H_L, att_tuple


def run_training(epochs, data):
    # TODO: move all this training code to the right method
    device = torch.device("cpu")

    in_channels, hidden_channels, out_channels = data.x.shape[1], 16, 8

    gae = GAE(gat_model.GATLayer(in_channels, hidden_channels, out_channels))

    gae = gae.to(device)
    gae = gae.float()

    data = data.to(device)

    optimizer = torch.optim.Adam(gae.parameters(), lr=0.01)

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

    return data, att_tuple
