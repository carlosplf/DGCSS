import networkx as nx
import torch
import numpy as np
from utils.utils import edges_to_edgeindex
from utils.graph_viewer import show_graph


def define_graph():
    # Creating a graph with DiGraph()
    G = nx.DiGraph()

    # Defining some community values by hand, just for testing
    communities = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2,
                   2, 2, 2, 2, 2]

    # Creating a Toy Graph
    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3),
             (2, 5), (3, 4), (3, 5), (4, 5), (5, 9), (6, 9),
             (6, 10), (6, 11), (7, 9), (7, 10), (7, 11),
             (8, 10), (8, 11), (8, 9),
             (8, 17),  (12, 13), (12, 14),
             (12, 15), (13, 14), (14, 15), (14, 17), (15, 16),
             (15, 17), (16, 17)]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G.add_edges_from([(v, u) for (u, v) in edges])

    edge_index = edges_to_edgeindex(list(G.edges))
    G.edge_index = edge_index

    print(len(edges))

    return G, communities


def run():
    G, communities = define_graph()

    # Adding some featires to the Graph
    X = torch.tensor(np.eye(18), dtype=torch.float)

    G.features = X

    for i in range(len(G.nodes())):
        G.nodes[i]['features'] = X[i]
        G.nodes[i]['label'] = communities[i]

    # Get nodes coordinates to visualize
    pos = nx.spring_layout(G, seed=42)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    show_graph(G)


if __name__ == "__main__":
    run()
