import networkx as nx
import itertools
from torch_geometric.utils import to_networkx


def run_gn(data, number_of_groups):
    """
    Execute the Girvan Newman algorithm and find the best communities
    set for the number of communities specified.
    Args:
        (Pytorch Data): data being processed 
        (int) number_of_groups
    Return:
        (tuple): communities definition
    """
    netxG = nx.Graph(to_networkx(data))
    G = nx.path_graph(netxG)

    k = number_of_groups
    
    comp = nx.community.girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    communities_final = None
    
    for communities in limited:
        communities_final = tuple(sorted(c) for c in communities)

    return communities_final
