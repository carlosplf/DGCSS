import warnings
import torch
import numpy as np
import logging
from torch_geometric.nn import GAE
from gat_model import gat_model
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import clustering_loss
from utils import csv_writer
from utils import plot_centroids
from centroids_finder import (
    random_seeds,
    fastgreedy,
    kcore,
    weighted_modularity,
    pagerank,
    kmeans,
    betweenness_centrality,
)


# Ignore torch FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

LEARNING_RATE = 0.01  # Learning rate
LR_CHANGE_GAMMA = 0.5  # Multiplier for the Learning Rate
LR_CHANGE_EPOCHS = 50  # Interval to apply LR change


class GaeRunner:
    def __init__(
        self,
        epochs,
        data,
        b_edge_index,
        n_clusters,
        find_centroids_alg,
        c_loss_gama,
        p_interval,
        centroids_plot_file,
    ):
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
        self.c_loss_gama = c_loss_gama
        self.p_interval = p_interval
        self.centroids_plot_file = centroids_plot_file

    def __print_values(self):
        logging.info("C_LOSS_GAMMA: " + str(self.c_loss_gama))
        logging.info("LEARNING_RATE: " + str(LEARNING_RATE))
        logging.info("CALC_P_INTERVAL: " + str(self.p_interval))
        logging.info("LR_CHANGE_GAMMA: " + str(LR_CHANGE_GAMMA))
        logging.info("LR_CHANGE_EPOCHS: " + str(LR_CHANGE_EPOCHS))

    def run_training(self):
        self.__print_values()

        # Check if CUDA is available and define the device to use.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("Running on " + str(device))

        in_channels, hidden_channels, out_channels = self.data.x.shape[1], 512, 16

        msg = (
            "Network structure: "
            + str(in_channels)
            + ", "
            + str(hidden_channels)
            + ", "
            + str(out_channels)
        )
        logging.info(msg)

        gae = GAE(gat_model.GATLayer(in_channels, hidden_channels, out_channels))

        gae = gae.float()

        # Move everything to the right device
        gae = gae.to(device)
        self.data = self.data.to(device)
        self.b_edge_index = self.b_edge_index.to(device)

        optimizer = torch.optim.Adam(gae.parameters(), lr=LEARNING_RATE)  # pyright: ignore
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_CHANGE_EPOCHS, gamma=LR_CHANGE_GAMMA
        )

        losses = []
        att_tuple = [[]]
        error_log = []
        Z = None

        for epoch in range(self.epochs):
            loss, Z, att_tuple, c_loss, gae_loss = self.__train_network(
                gae, optimizer, epoch, scheduler
            )
            logging.info("==> " + str(epoch) + " - Loss: " + str(loss))
            losses.append(loss)
            error_log.append([epoch, loss, c_loss, gae_loss])

        r = []

        for line in self.Q:  # pyright: ignore
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
            plot_centroids.plot_centroids(
                Z, self.clusters_centroids, self.centroids_plot_file
            )

        self.Q = clustering_loss.calculate_q(self.clusters_centroids, Z)

        if epoch % self.p_interval == 0:
            self.P = clustering_loss.calculate_p(self.Q)

        Lc, Q, P = clustering_loss.kl_div_loss(self.Q, self.P)

        gae_loss = gae.recon_loss(Z, self.data.edge_index)

        total_loss = gae_loss + self.c_loss_gama * Lc

        total_loss.backward()

        if self.first_interaction is False and Lc != 0:
            self.clusters_centroids = clustering_loss.update_clusters_centers(
                self.clusters_centroids, Q.grad
            )

        optimizer.step()
        scheduler.step()

        self.first_interaction = False

        return float(total_loss), Z, att_tuple, float(Lc), float(gae_loss)

    def _find_centroids(self, Z):
        """
        Find the centroids using the selected algorithm.
        Args:
            Z: The matrix representing the nodes AFTER the Encoding process.
        """

        if self.find_centroids_alg == "WFastGreedy":
            self.clusters_centroids = weighted_modularity.select_centroids(
                self.data, Z, "weight"
            )

        elif self.find_centroids_alg == "Random":
            self.clusters_centroids = random_seeds.select_centroids(Z)

        elif self.find_centroids_alg == "BC":
            self.clusters_centroids = betweenness_centrality.select_centroids(
                self.data, Z
            )

        elif self.find_centroids_alg == "PageRank":
            self.clusters_centroids = pagerank.select_centroids(self.data, Z)

        elif self.find_centroids_alg == "KMeans":
            self.clusters_centroids = kmeans.select_centroids(Z, self.n_clusters)

        elif self.find_centroids_alg == "FastGreedy":
            self.clusters_centroids = fastgreedy.select_centroids(self.data, Z)

        elif self.find_centroids_alg == "KCore":
            self.clusters_centroids = kcore.select_centroids(self.data, Z)

        else:
            logging.error("FIND_CENTROIDS_ALG not known. Aborting...")
            self.clusters_centroids = []

        log_msg = "Centroids: " + str(self.clusters_centroids)
        logging.debug(log_msg)
