import heapq
import logging
import networkx as nx
from torch_geometric.utils import to_networkx
from utils import distance_calculator


def select_centroids(data=None, Z=None, n_clusters=0):
    logging.info("Using Closeness Centrality to find the centroids...")

    G = nx.Graph(to_networkx(data, node_attrs=["x"]))
   
    bc_nodes = nx.closeness_centrality(G)
    
    biggest = heapq.nlargest(n_clusters, bc_nodes.items(), key=lambda i: i[1])
    centroids = [i[0] for i in biggest]

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids

