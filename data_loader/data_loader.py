import networkx as nx
import numpy as np
from torch_geometric.utils.convert import from_networkx


def load_as_graph(dataset_name="easy_small", root_dir='data_graph_exp/',
                  graph_index=0):

    loaded = np.load(root_dir+dataset_name+'.npz', allow_pickle=True)

    # Train Dataset and Graphs
    A_train = list(loaded['tr_adj'])
    X_train = loaded['tr_feat']
    # Y_train = loaded['tr_class']

    # Test Dataset and Graphs
    # As we are going for a unsupervised approach, we don't
    # need those values.
    # A_test = list(loaded['te_adj'])
    # X_test = loaded['te_feat']
    # Y_test = loaded['te_class']

    # Convert to networkx format
    G_tr = []
    for a, x in zip(A_train, X_train):
        G = nx.from_scipy_sparse_array(a, create_using=nx.DiGraph)
        x_tuple = tuple(map(tuple, x))
        nx.set_node_attributes(G, dict(enumerate(x_tuple)), 'features')
        G_tr.append(G)

    # G_te = []
    # for a, x in zip(A_test, X_test):
    #     G = nx.from_scipy_sparse_array(a, create_using=nx.DiGraph)
    #     x_tuple = tuple(map(tuple, x))
    #     nx.set_node_attributes(G, dict(enumerate(x_tuple)), 'features')
    #     G_te.append(G)

    data_as_pytorch_dataset = from_networkx(G_tr[graph_index])
    return data_as_pytorch_dataset, X_train[graph_index]
