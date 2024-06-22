import torch
import numpy as np
import networkx as nx
import logging
from torch_geometric.nn import GAE
from gat_model import gat_model
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import clustering_loss
from utils import fast_greedy
from utils import csv_writer
from utils import plot_centroids
from torch_geometric.utils import to_networkx


C_LOSS_GAMMA = 6
LEARNING_RATE = 0.01
CALC_P_INTERVAL = 5
LR_CHANGE_GAMMA = 0.5


class GaeRunner:
    def __init__(self, epochs, data, b_edge_index, n_clusters,
                 find_centroids_alg):
        self.epochs = epochs
        self.data = data
        self.b_edge_index = b_edge_index
        self.n_clusters = n_clusters
        self.Q = 0
        self.P = 0
        self.clusters_centroids = None
        self.first_interaction = True
        self.communities = None
        self.mod_score = None
        self.find_centroids_alg = find_centroids_alg
        self.error_log_filename = "error_log.csv"

    def run_training(self):
        # Check if CUDA is available and define the device to use.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("Running on " + str(device))

        in_channels, hidden_channels, out_channels = \
            self.data.x.shape[1], 256, 16

        gae = GAE(gat_model.GATLayer(in_channels, hidden_channels,
                                     out_channels))

        gae = gae.float()

        # Move everything to the right device
        gae = gae.to(device)
        self.data = self.data.to(device)
        self.b_edge_index = self.b_edge_index.to(device)

        optimizer = torch.optim.Adam(gae.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40,
                                                    gamma=LR_CHANGE_GAMMA)

        losses = []
        att_tuple = [[]]
        error_log = []
        Z = None

        for epoch in range(self.epochs):
            loss, Z, att_tuple = self.__train_network(gae, optimizer, epoch,
                                                      scheduler)
            logging.info("==> " + str(epoch) + " - Loss: " + str(loss))
            losses.append(loss)
            error_log.append([epoch, loss])

        r = []
        for line in self.Q:
            r.append(np.argmax(line))

        logging.info(
            "Normalized mutual info score: "
            + str(normalized_mutual_info_score(self.data.y.tolist(), r))
        )
        
        csv_writer.write_erros(error_log, self.error_log_filename)

        return self.data, att_tuple

    def __train_network(self, gae, optimizer, epoch, scheduler):
        gae.train()
        optimizer.zero_grad()

        att_tuple, Z = gae.encode(
            self.data.x.float(),
            self.b_edge_index.edge_index,
            self.b_edge_index.edge_attr,
        )

        if self.clusters_centroids is None:
            self._find_centroids(Z)

        plot_centroids.plot_centroids(Z, self.clusters_centroids)

        self.Q = clustering_loss.calculate_q(self.clusters_centroids, Z)

        if epoch % CALC_P_INTERVAL == 0:
            self.P = clustering_loss.calculate_p(self.Q)

        Lc, Q, P = clustering_loss.kl_div_loss(self.Q, self.P)

        gae_loss = gae.recon_loss(Z, self.data.edge_index)

        total_loss = gae_loss + C_LOSS_GAMMA*Lc

        total_loss.backward()

        if self.first_interaction is False and Lc != 0:
            self.clusters_centroids = clustering_loss.update_clusters_centers(
                self.clusters_centroids, Q.grad)

        optimizer.step()
        scheduler.step()

        self.first_interaction = False

        return float(total_loss), Z, att_tuple

    def _find_centroids(self, Z):
        """
        Find the centroids using the selected algorithm.
        """

        if self.find_centroids_alg == "KMeans":
            logging.info("Using KMeans to find the centroids...")
            # First time running the treining, calculate centroids using KMeans
            self.clusters_centroids = clustering_loss.get_clusters_centroids(
                Z, self.n_clusters
            )

        elif self.find_centroids_alg == "FastGreedy":
            logging.info("Using Fast Greedy to find the centroids...")
            G = nx.Graph(to_networkx(self.data, node_attrs=['x']))

            # First time running the treining, find communities
            # using Fast Greedy
            self.communities = fast_greedy.run_fast_greedy(G, 5, 5)

            # For each community, find the centroid
            centroids = fast_greedy.get_clusters_centroids(G, self.communities)

            self.clusters_centroids = []

            # Get Z values for each centroid.
            for c in centroids:
                self.clusters_centroids.append(Z[c].tolist())

        else:
            logging.error("FIND_CENTROIDS_METHOD not known. Aborting...")
            self.clusters_centroids = []

        logging.debug(self.clusters_centroids)
