import networkx as nx
from utils.utils import edges_to_edgeindex
import torch_geometric.utils.convert as convert
import torch


def define_graph():
    # Creating a graph with DiGraph()
    G = nx.DiGraph()

    # Defining some community values by hand, just for testing
    communities = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2,
                   2, 2, 2, 2, 2]

    # Creating a Toy Graph just to use as an example
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


def create_from_dataset(data):
    # Creating a graph with DiGraph()
    G = nx.DiGraph()

    # Defining some community values by hand, just for testing
    communities = data.y

    # Creating a Toy Graph just to use as an example
    nodes = range(0, data.num_nodes)
    edges = build_edges_tuples(data)

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G.add_edges_from([(v, u) for (u, v) in edges])

    edge_index = edges_to_edgeindex(list(G.edges))
    G.edge_index = edge_index

    return G, communities


def build_edges_tuples(data):
    new_edge_indexes = []
    edges_tensor_as_list = data.edge_index.tolist()
    for i in range(len(edges_tensor_as_list[0])):
        new_edge_indexes.append(
            (edges_tensor_as_list[0][i],
             edges_tensor_as_list[1][i])
            )
    return new_edge_indexes
