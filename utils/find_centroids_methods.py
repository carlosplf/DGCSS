import heapq
import networkx as nx


def by_pagerank(G, n_centroids=5):
    """
    Find centroids using the PageRank mechanism.
    Ref.: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
    Args:
        G (networkx Graph) Graph to consider.
    Return:
        (list) List with centroids IDs.
    """

    page_rank_nodes = nx.pagerank(G, alpha=0.9)
    biggest = heapq.nlargest(n_centroids, page_rank_nodes.items(), key=lambda i: i[1])
    return [i[0] for i in biggest]