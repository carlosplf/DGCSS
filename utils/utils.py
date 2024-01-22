import torch
import networkx as nx


def edges_to_edgeindex(edges):
    '''
    Transforma uma lista de edge index do formato
    [(x1,y1), (x2,y2)] no formato [[x1,x2], [y1,y2]]
    '''
    src = [x[0] for x in edges]
    tgt = [x[1] for x in edges]
    return torch.tensor([src, tgt])


def remove_min_weight_edges(graph):
    '''
    Remove edges with less weights.
    '''
    # Encontre o peso mínimo das arestas
    min_weight = min(graph.edges(data=True),
                     key=lambda x: x[2]['weight'])[-1]['weight']

    # Encontre todas as arestas com o peso mínimo
    edges_to_remove = [
        (u, v) for u, v, data in graph.edges(data=True)
        if data['weight'] == min_weight]

    # Remova as arestas do grafo
    graph.remove_edges_from(edges_to_remove)

    return graph


# Transforma a tupla em matriz de adjacência (somente para visualização)
def tuple_to_adj(att_tuple, G):
    # att_tuple[1] -> lista de pesos de cada aresta no grafo
    # att_tuple[0][0] -> lista de source de cada aresta
    # att_tuple[0][1] -> lista de target de cada aresta
    adj = torch.zeros((len(G.nodes()), len(G.nodes())))
    for i in range(len(att_tuple[1])):
        adj[att_tuple[0][0][i], att_tuple[0][1][i]] = att_tuple[1][i]

    return adj, att_tuple[1]


def remove_edges(G, communities, num_edges_to_remove=None):
    # Remove weights with small weights, based on the Attention values.
    print("Removing edges with small Attention values...")

    num_rem = 0
    if not num_edges_to_remove:
        while nx.number_connected_components(G.to_undirected()) != 3:
            G = remove_min_weight_edges(G)
            num_rem += 1
    else:
        for i in range(num_edges_to_remove):
            G = remove_min_weight_edges(G)
            num_rem += 1

    print("Removed", num_rem, "edges.")

    return G, communities
