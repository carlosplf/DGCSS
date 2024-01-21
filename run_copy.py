import networkx as nx
import torch
import numpy as np
from utils.utils import edges_to_edgeindex
from utils.utils import remove_min_weight_edges
from utils.graph_viewer import show_graph
from utils.graph_viewer import plot_weights
from utils.graph_creator import define_graph
from utils.graph_creator import create_from_dataset
from gat_model import gat_model
import torch_geometric.utils as utils
from torch_geometric.nn import GAE
import random
from torch_geometric.datasets import Planetoid


# Defining random seeds
random.seed(81)
np.random.seed(81)
torch.manual_seed(81)
torch.cuda.manual_seed(81)


def import_dataset():
    # Defina o diretório onde você deseja armazenar o conjunto de dados
    root = './data/Cora'

    # Baixe o conjunto de dados Cora e o carregue
    dataset = Planetoid(root=root, name='Cora', transform=None, pre_transform=None)

    # Imprima algumas informações sobre o conjunto de dados
    print('Número de gráficos (grafos):', len(dataset))
    print('Número de classes:', dataset.num_classes)
    print('Número de recursos:', dataset.num_node_features)

    # Acesse o primeiro gráfico no conjunto de dados
    data = dataset[0]

    # Imprima algumas informações sobre o primeiro gráfico
    print('\nInformações sobre o primeiro gráfico:')
    print(data)

    return data


def train_network(gae, optimizer, graph):
    gae.train()
    optimizer.zero_grad()

    att_tuple, H_L = gae.encode(graph.features.float(), graph.edge_index)

    # Decode por multiplicação pela transposta
    loss = gae.recon_loss(H_L, graph.edge_index)

    return float(loss), H_L, att_tuple


def run(data):

    G, communities = create_from_dataset(data)

    # Adding some features to the Graph
    X = data.x

    # Duplicated operation?
    G.features = data.x

    for i in range(len(G.nodes())):
        G.nodes[i]['features'] = X[i]
        G.nodes[i]['label'] = communities[i]

    device = torch.device('cpu')

    dataset = utils.from_networkx(G)

    in_channels, hidden_channels, out_channels = len(dataset.features[0]), 8, 2

    gae = GAE(gat_model.GATLayer(in_channels, hidden_channels, out_channels))

    gae = gae.to(device)
    gae = gae.float()

    dataset = dataset.to(device)

    optimizer = torch.optim.Adam(gae.parameters(), lr=0.005)

    epochs = 200

    losses = []
    embs_list = []

    for epoch in range(epochs):
        loss, H_L, att_tuple = train_network(gae, optimizer, dataset)
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

    G, communities = remove_edges(G, communities)
    plot_weights(G, communities)


def remove_edges(G, communities):
    # Remove weights with small weights, based on the Attention values.
    num_rem = 0
    to_remove = round(len(G.edges(data=True))/2)
    for i in range(0, to_remove):
        G = remove_min_weight_edges(G)
        num_rem += 1

    print("Removed", num_rem, "edges.")

    return G, communities


if __name__ == "__main__":
    data = import_dataset()
    run(data)
