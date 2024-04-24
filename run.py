import argparse
import logging

import numpy as np
from numpy.linalg import matrix_power
import torch
from torch_geometric.data import Data


from runners import gae_runner
from utils.graph_creator import get_cora_dataset
from sklearn.cluster import AgglomerativeClustering


parser = argparse.ArgumentParser()

parser.add_argument(
    "--epochs", type=int, help="Define number of EPOCHS for training.", default=10
)


GRAPH_NUMBER_NODES = 250
GRAPH_NUMBER_CLASSES = 5


def run_aggl_clustering(att_tuple):
    """
    Run the AgglomerativeClustering algorithm and try to classify the node.
    Args:
        att_tuple: [[]] Attention values from Graph Autoencoder.
    Return: None
    """
    new_att = np.zeros((GRAPH_NUMBER_NODES, GRAPH_NUMBER_NODES))

    src = att_tuple[0][0].detach().numpy()
    tgt = att_tuple[0][1].detach().numpy()
    weight = att_tuple[1].detach().numpy()

    # TODO: attention matrix should be about distances, not values
    for i in range(len(src)):
        new_att[src[i]][tgt[i]] = 1/weight[i]

    aggl_cl = AgglomerativeClustering(linkage="complete", n_clusters=5, metric='precomputed')
    aggl_result = aggl_cl.fit_predict(X=new_att)
    return aggl_result


def calc_t_order_neighbors(adj_matrix, t=1):
    """
    Calculate the T order neighbors for the given adjancency matrix.
    """
    b = np.zeros((GRAPH_NUMBER_NODES, GRAPH_NUMBER_NODES))
    for i in range(GRAPH_NUMBER_NODES):
        for j in range(GRAPH_NUMBER_NODES):
            if adj_matrix[i][j] == 0:
                b[i][j] = 0
                continue
            sum_b_row = np.sum(adj_matrix[i])
            b[i][j] = 1/sum_b_row
    
    b_sum = b
    for i in range(1, t):
        b_sum = b_sum + matrix_power(b, i)
        
    return b_sum/t


def create_adj_matrix(data):
    new_adj_matrix = np.zeros((data.num_nodes, data.num_nodes))

    src = data.edge_index[0].detach().numpy()
    tgt = data.edge_index[1].detach().numpy()

    for i in range(len(src)):
        new_adj_matrix[src[i]][tgt[i]] = 1

    return new_adj_matrix


def adj_to_edge_index(adj_matrix):
    """
    Transform a adjacency matrix to the Edge Index format.
    """
    adj_t = torch.tensor(adj_matrix)
    edge_index = adj_t.nonzero().t().contiguous()
    
    return edge_index


def run(epochs):

    data = get_cora_dataset()

    adj_matrix = create_adj_matrix(data)

    b_matrix = calc_t_order_neighbors(adj_matrix=adj_matrix, t=4)
    
    b_src_tgt_tensor = adj_to_edge_index(b_matrix)

    # After build the edge_index format, calculate the weights.
    b_weights = []
    
    for i in range(len(b_src_tgt_tensor[0])):
        b_weights.append(adj_matrix[b_src_tgt_tensor[0][i], b_src_tgt_tensor[1][i]])

    b_weights = torch.tensor(b_weights)
    
    b_edge_index = Data(edge_index=b_src_tgt_tensor, edge_attr=b_weights)

    runner = gae_runner.GaeRunner(epochs, data, b_edge_index, GRAPH_NUMBER_CLASSES)

    data, att_tuple = runner.run_training()

    return


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()

    epochs = args.epochs

    logger.info("Considering %s epochs", epochs)

    run(epochs)
