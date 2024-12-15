import heapq
import logging
import networkx as nx
from torch_geometric.utils import to_networkx
from utils import distance_calculator


def select_centroids(data, Z, n_clusters):
    logging.info("Using Weighted Betweenness Centrality to find the centroids...")
    G = nx.Graph(to_networkx(data, node_attrs=["x"]))
   
    distances = distance_calculator.graph_manhattan_distances(Z.detach().numpy())
    G = distance_calculator.define_weights(G, distances, "weight")

    bc_nodes = nx.betweenness_centrality(G, weight="weight")
    biggest = heapq.nlargest(n_clusters, bc_nodes.items(), key=lambda i: i[1])
    centroids = [i[0] for i in biggest]

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids
