# TODO: recheck imports

import torch
import time
from gat_model import gat_model
from torch_geometric.logging import log
import torch.nn.functional as F
from utils.graph_creator import create_from_dataset
from utils.utils import remove_edges
# from utils.graph_viewer import plot_weights
import networkx as nx


@torch.no_grad()
def test(model, data):
    model.eval()
    att_tuple, x = model(data.x, data.edge_index)
    pred = x.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def train_network_gat(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    att_tuple, out = model(data.x, data.edge_index)

    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return float(loss), att_tuple


def run_training(dataset, epochs):

    device = torch.device('cpu')
    data = dataset[0].to(device)

    in_channels, hidden_channels, out_channels = \
        dataset.num_features, 8, dataset.num_classes

    model = gat_model.GAT_test2(
        in_channels, hidden_channels, out_channels,
        heads=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005,
                                 weight_decay=5e-4)

    times = []
    best_val_acc = 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        loss, att_tuple = train_network_gat(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss,
            Train=train_acc, Val=val_acc, Test=test_acc)
        times.append(time.time() - start)

    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    G, communities = build_graph(dataset)
    remove_edges_and_plot_graph(G, communities, att_tuple)


def build_graph(dataset):
    data = dataset[0]
    G, communities = create_from_dataset(data)
    X = data.x

    G.features = X

    for i in range(len(G.nodes())):
        G.nodes[i]['features'] = X[i]
        G.nodes[i]['label'] = communities[i]

    return G, communities


def remove_edges_and_plot_graph(G, communities,
                                att_tuple, dataset_to_use="Cora"):

    total_edges = G.number_of_edges()
    print("Total number of edges on Graph:", total_edges)
    edges_to_remove = round(total_edges/4)

    # Add the Attention values to the original Graph edges
    weight = att_tuple[1]
    src = att_tuple[0][0]
    tgt = att_tuple[0][1]

    for i in range(len(weight)):
        G.add_edge(src[i].item(), tgt[i].item(), weight=weight[i].item())

    # Plot original graph with edge weights
    # plot_weights(G, communities)

    print("Clustering coefficient:", nx.average_clustering(G))

    if dataset_to_use == "Cora":
        G, communities = remove_edges(G, communities,
                                      num_edges_to_remove=edges_to_remove)
    else:
        G, communities = remove_edges(G, communities)

    # plot_weights(G, communities)
    print("Clustering coefficient:", nx.average_clustering(G))
