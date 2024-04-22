# TODO: recheck imports

import torch
import numpy as np
import logging
from torch_geometric.nn import GAE
from gat_model import gat_model
from sklearn.cluster import KMeans


def get_clusters_centroids(Z, n_clusters):
    """
    Runs KMeans clustering to find the centroids.
    """
    logging.info("Calculating centroids with K-Means...")
    X = Z.detach().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    return kmeans.cluster_centers_


def calculate_q(clusters_centroids, Z):
    """
    Calculate Q using Student’s t-distribution.
    """

    if clusters_centroids is None:
        return None
    
    nodes = Z.detach().numpy()
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

    if Q is None:
        return None
    
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

    if P is None or Q is None:
        return 1000
    
    number_of_nodes = len(Q)
    number_of_centroids = len(Q[0])
    
    loss_clustering = 0

    for i in range(number_of_nodes):
        for u in range(number_of_centroids):
            loss_clustering += (P[i][u] * (np.log(P[i][u]/Q[i][u])))

    return loss_clustering


def train_network_gae(epoch, gae, optimizer, data, b_edge_index,
                      n_clusters, clusters_centroids):
    gae.train()
    optimizer.zero_grad()

    att_tuple, H_L = gae.encode(
        data.x.float(), b_edge_index.edge_index, b_edge_index.edge_attr
    )

    if clusters_centroids is None:
        clusters_centroids = get_clusters_centroids(H_L, n_clusters)

    Q = calculate_q(clusters_centroids, H_L)
    P = calculate_p(Q)
    loss_clustering = clustering_loss(Q, P)
    logging.info("==> Loss clustering: " + str(loss_clustering))

    y = 0.1
    
    loss = gae.recon_loss(H_L, data.edge_index)
    
    total_loss = loss + y*loss_clustering

    total_loss.backward()
    optimizer.step()

    return float(total_loss), H_L, att_tuple, clusters_centroids


def run_training(epochs, data, b_edge_index, n_clusters):
    # TODO: move all this training code to the right method
    device = torch.device("cpu")

    in_channels, hidden_channels, out_channels = data.x.shape[1], 64, 32

    gae = GAE(gat_model.GATLayer(in_channels, hidden_channels, out_channels))

    gae = gae.to(device)
    gae = gae.float()

    data = data.to(device)

    optimizer = torch.optim.Adam(gae.parameters(), lr=0.001)

    losses = []
    embs_list = []
    att_tuple = [[]]

    clusters_centroids = None

    for epoch in range(epochs):
        loss, H_L, att_tuple, clusters_centroids = train_network_gae(
            epoch, gae, optimizer, data, b_edge_index, n_clusters, clusters_centroids
        )
        if epoch % 10 == 0:
            logging.info("==> " + str(epoch) + " - Loss: " + str(loss))
        losses.append(loss)
        embs_list.append(H_L)

    # plt.plot(losses, label="train_loss")
    # plt.show()

    return data, att_tuple
