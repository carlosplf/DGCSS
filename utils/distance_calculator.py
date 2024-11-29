import logging
import networkx as nx
from sklearn.metrics.pairwise import manhattan_distances


def graph_manhattan_distances(X, consider_edges=False, mechanism="manhattan"):
    """
    Calculate the manhattan distance from all N to N nodes in a graph.

    Args:
        X ([[]]): The nodes attributes values.

        consider_edges (bool): If True, distance between nodes that doesn't
        have edge between then is infinite. If False, ignore edges in the
        and calculate based on the attributes.

        mechanism (str): the mechanism to be used to calculate the distances.

    Return:

        Return a matrix N to N, where N is the number of nodes, containing
        all the distances between nodes.

    """
    number_of_nodes = len(X)
    distances = [([0] * number_of_nodes) for i in range(number_of_nodes)]

    logging.info("Calculating distances between nodes.")
    logging.info("Number of nodes: " + str(number_of_nodes))
    logging.info("Distance mechanism: " + mechanism)

    # TODO: Implement other mechanisms to calculate distance.
    if mechanism == "manhattan":
        distances = manhattan_distances(X, X)
    else:
        logging.error("Distance mechanism not known. Aborting...")
        return None

    return distances


def define_weights(G, distances, weight_name):
    """
    Define the weights in the Graph based on the distance between nodes.
    The weight is defined by 1/<distance>.
    Args:
        G (networkx G): Original Graph.
        distances ([[]]): Matrix NxN with all the distances between nodes.
        weight_name (str): Name of the field that will be stored in the Graph.
    Return:
        New networkx Graph with weights.
    """
    g_attrs = {}

    number_of_nodes = len(distances)

    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if distances[i][j] == 0:
                weight = 1
            else:
                weight = 1 / distances[i][j]
            g_attrs[(i, j)] = {weight_name: weight}

    nx.set_edge_attributes(G, g_attrs)
    return G
