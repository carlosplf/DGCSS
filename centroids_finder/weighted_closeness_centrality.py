import heapq
import logging
import networkx as nx
from torch_geometric.utils import to_networkx
from utils import distance_calculator


def select_centroids(data=None, Z=None, n_clusters=0):
    logging.info("Using Closeness Centrality with Weights to find the centroids...")

    G = nx.Graph(to_networkx(data, node_attrs=["x"]))
   
    # Calculating diatances via COSINE SIMILARITY. 1 = equal, 0 = very different.
    # Distances are calculated based on Attributes. It can be X or Z.

    # Based on Graph X (attributes).
    logging.info("Calculating distances based on X.")
    distances = distance_calculator.graph_attr_distances(data.x, mechanism="cosine")

    g_attrs = distance_calculator.define_weights(
        G=G, distances=distances, weight_name="distancia", multiplier="inverse"
    )

    nx.set_edge_attributes(G, g_attrs)
    
    # Weights are used to calculate weighted shortest paths, so they are interpreted as distances
    bc_nodes = nx.closeness_centrality(G, distance="distancia")
    
    
    biggest = heapq.nlargest(n_clusters, bc_nodes.items(), key=lambda i: i[1])
    centroids = [i[0] for i in biggest]

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids

