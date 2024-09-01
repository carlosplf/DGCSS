import logging
from numpy import number

from sklearn.metrics.pairwise import manhattan_distances


def graph_manhattan_distances(G, X, consider_edges=False, mechanism="manhattan"):
    """
    Calculate the manhattan distance from all N to N nodes in a graph.
    
    Args:
        
        G (Networkx graph format): The graph to consider.

        X ([[]]): The nodes attributes values.
        
        consider_edges (bool): If True, diatance between nodes that doesn't
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

    if mechanism is "manhattan":
        distances = manhattan_distances(X, X)
    else:
        logging.error("Distance mechanism not known. Aborting...")
        return None

    return distances
