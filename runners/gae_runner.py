import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch_geometric.nn import GAE
from gat_model import gat_model
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import clustering_loss


C_LOSS_GAMA = 12
LEARNING_RATE = 0.01
CALC_P_INTERVAL = 10


class GaeRunner():

    def __init__(self, epochs, data, b_edge_index, n_clusters):
        self.epochs = epochs
        self.data = data
        self.b_edge_index = b_edge_index
        self.n_clusters = n_clusters
        self.Q = 0
        self.P = 0
        self.clusters_centroids = None
        self.first_interaction = True

    def run_training(self):

        # Check if CUDA is available and define the device to use.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("Running on " + str(device))

        in_channels, hidden_channels, out_channels = self.data.x.shape[1], 256, 16

        gae = GAE(gat_model.GATLayer(in_channels, hidden_channels, out_channels))

        gae = gae.float()

        # Move everything to the right device
        gae = gae.to(device)
        self.data = self.data.to(device)
        self.b_edge_index = self.b_edge_index.to(device)

        optimizer = torch.optim.Adam(gae.parameters(), lr=LEARNING_RATE)

        losses = []
        att_tuple = [[]]

        for epoch in range(self.epochs):
            loss, Z, att_tuple = self.__train_network(gae, optimizer, epoch)
            if epoch % 10 == 0:
                logging.info("==> " + str(epoch) + " - Loss: " + str(loss))
            losses.append(loss)

        r = []
        for line in self.Q:
            r.append(np.argmax(line))

        logging.info("Normalized mutual info score: " + str(normalized_mutual_info_score(self.data.y.tolist(), r)))

        return self.data, att_tuple

    def __train_network(self, gae, optimizer, epoch):

        gae.train()
        optimizer.zero_grad()

        att_tuple, Z = gae.encode(
            self.data.x.float(), self.b_edge_index.edge_index, self.b_edge_index.edge_attr
        )

        if self.clusters_centroids is None:
            # First time running the treining, calculate centroids using KMeans
            self.clusters_centroids = clustering_loss.get_clusters_centroids(Z, self.n_clusters)

        self.Q = clustering_loss.calculate_q(self.clusters_centroids, Z)

        if epoch % CALC_P_INTERVAL == 0:
            self.P = clustering_loss.calculate_p(self.Q)

        # loss_clustering = clustering_loss.calculate_clustering_loss(self.Q, self.P)

        # transforming into Tensor to have gradients and backwards calculation.
        Q = torch.tensor(self.Q, requires_grad=True)
        P = torch.tensor(self.P, requires_grad=True)

        loss_clustering = F.kl_div(Q.log(), P)
        
        logging.info("Lc: " + str(loss_clustering))

        gae_loss = gae.recon_loss(Z, self.data.edge_index)

        total_loss = gae_loss + C_LOSS_GAMA*loss_clustering

        total_loss.backward()
        
        if self.first_interaction is False and loss_clustering != 0:
            self.clusters_centroids = clustering_loss.update_clusters_centers(self.clusters_centroids, Q.grad)

        optimizer.step()

        self.first_interaction = False

        return float(total_loss), Z, att_tuple
