import networkx as nx


def run_fast_greedy(G, cutoff, best_n):
    """
    Runs the Fast Greedy algorithn.
    Source: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html
    Return:
        (list): List of all communities.
    """
    comms = nx.community.greedy_modularity_communities(G, cutoff=cutoff, best_n=best_n)
    return comms


def get_clusters_centroids(G, communities):
    """
    Return the node that represents the centroid for each cluster.
    The centroid is chosen based on the centrality degrees of the node.
    """
   
    # TODO: Recheck this method.
    centrality_degrees = nx.degree_centrality(G)
    best_node_centrality = 0
    best_nodes = [0] * len(communities)
    comm_idx = 0

    for community in communities:
        for node in community:
            if centrality_degrees[node] > best_node_centrality:
                best_node_centrality = centrality_degrees[node]
                best_nodes[comm_idx] = node
        comm_idx += 1
        best_node_centrality = 0

    return best_nodes

