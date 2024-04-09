# TODO: recheck imports

import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx, to_edge_index, dense_to_sparse
from torch_geometric.nn import GAE
from gat_model import gat_model
from sklearn.cluster import KMeans


def get_clusters_centroids(data, n_clusters):
    """
    Runs KMeans clustering to find the centroids.
    """
    X = data.x.detach().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    return kmeans.cluster_centers_


def calculate_q(clusters_centroids, data):
    """
    Calculate Q using Student’s t-distribution.
    """
    nodes = data.x.detach().numpy()
    number_of_nodes = len(nodes)
    number_of_centroids = len(clusters_centroids)

    Q = np.zeros((number_of_nodes, number_of_centroids))

    for i in range(number_of_nodes):
        for u in range(number_of_centroids):
            q = 1 / ( 1 + np.power(np.linalg.norm(nodes[i]-clusters_centroids[u]), 2) )
            q_lower_sum = 0
            for k in range(number_of_centroids):
                q_lower_sum += 1 / ( 1 + np.power(np.linalg.norm(nodes[i]-clusters_centroids[k]), 2) )
            q = q / q_lower_sum
            Q[i][u] = q        

    return Q


def calculate_p(Q):
    """
    Calculate P using Q.
    """
    
    number_of_nodes = len(Q)
    number_of_centroids = len(Q[0])

    P = np.zeros((number_of_nodes, number_of_centroids))

    for i in range(number_of_nodes):
        for u in range(number_of_centroids):
            p = np.power(Q[i][u], 2) / sum(Q[x][u] for x in range(number_of_nodes))
            p_lower_sum = 0
            for k in range(number_of_centroids):
                p_lower_sum += (np.power(Q[i][k], 2) / sum(Q[x][k] for x in range(number_of_nodes)))
            p = p / p_lower_sum
            P[i][u] = p        
    return P


def clustering_loss(Q, P):
    number_of_nodes = len(Q)
    number_of_centroids = len(Q[0])
    
    loss_clustering = 0

    for i in range(number_of_nodes):
        for u in range(number_of_centroids):
            loss_clustering += (P[i][u] * (np.log(P[i][u]/Q[i][u])))

    print(loss_clustering)
    return loss_clustering


def train_network_gae(gae, optimizer, data, b_edge_index, clusters_centroids):
    gae.train()
    optimizer.zero_grad()

    att_tuple, H_L = gae.encode(
        data.x.float(), b_edge_index.edge_index, b_edge_index.edge_attr
    )

    Q = calculate_q(clusters_centroids, data)
    P = calculate_p(Q)
    loss_clustering = clustering_loss(Q, P)

    # Trying to calculate the Graph Edit Distance between the
    # original graph and the one that GEA built.
    # G_1 = to_networkx(data)
    
    # G_2 = nx.empty_graph(0, nx.DiGraph)
    # G_2.add_nodes_from(range(H_L.shape[0]))
    # G_2.add_edges_from(((int(e[0]), int(e[1])) for e in zip(*H_L.nonzero())))

    # Using too much memory!!!
    # ged = nx.graph_edit_distance(G_1, G_2)
    # print(ged)

    loss = gae.recon_loss(H_L, data.edge_index)

    loss.backward()
    optimizer.step()

    return float(loss), H_L, att_tuple


def run_training(epochs, data, b_edge_index, n_clusters):
    # TODO: move all this training code to the right method
    device = torch.device("cpu")

    in_channels, hidden_channels, out_channels = data.x.shape[1], 16, 8

    # Clusters centroids are calculated just once at the begining.
    clusters_centroids = get_clusters_centroids(data, n_clusters)

    gae = GAE(gat_model.GATLayer(in_channels, hidden_channels, out_channels))

    gae = gae.to(device)
    gae = gae.float()

    data = data.to(device)

    optimizer = torch.optim.Adam(gae.parameters(), lr=0.001)

    losses = []
    embs_list = []
    att_tuple = [[]]

    for epoch in range(epochs):
        loss, H_L, att_tuple = train_network_gae(
            gae, optimizer, data, b_edge_index, clusters_centroids
        )
        if epoch % 10 == 0:
            print("==>", epoch, "- Loss:", loss)
        losses.append(loss)
        embs_list.append(H_L)

    # plt.plot(losses, label="train_loss")
    # plt.show()

    return data, att_tuple
