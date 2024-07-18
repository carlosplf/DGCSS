import heapq
import networkx as nx


def find_centroids(G, number_of_centroids):
    """
    Return the nodes considered centroids by k-core ranking.
    "A k-core is a maximal subgraph that contains nodes of degree k or more".
    More info:
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.core_number.html
    """

    # This will return the k-core for each node.
    # {node_id: k-core, node_id: k-core, ...}
    core_numbers = nx.core_number(G)

    # Get the N nodes with the highest k-core score.
    highest_k_cores = heapq.nlargest(
        number_of_centroids,
        core_numbers.keys(),
        key=lambda k: core_numbers[k]
    )

    return highest_k_cores
