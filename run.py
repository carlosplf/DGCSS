import argparse
import random
import logging

import numpy as np
from numpy.linalg import matrix_power
import torch
from torch_geometric.utils import to_networkx, to_edge_index, dense_to_sparse
from torch_geometric.data import Data


from runners import gae_runner
from utils.graph_viewer import plot_weights
from utils.utils import remove_edges
from utils.graph_creator import create_demo_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score


parser = argparse.ArgumentParser()

parser.add_argument(
    "--epochs", type=int, help="Define number of EPOCHS for training.", default=10
)


GRAPH_NUMBER_NODES = 100
GRAPH_NUMBER_CLASSES = 4


def save_plots(graph_index, G_before, G_after, labels):
    logger.info("Saving graph images to folder.")
    
    filename = "before-" + str(graph_index) + ".png"
    plot_weights(G_before, labels, folder_path="./charts", filename=filename)

    filename = "after-" + str(graph_index) + ".png"
    plot_weights(G_after, labels, folder_path="./charts", filename=filename)


def set_attention_as_weights(data, att_tuple):
    # Add the Attention values to the original Graph edges
    weight = att_tuple[1]
    src = att_tuple[0][0]
    tgt = att_tuple[0][1]

    # After running all the training, transform the dataset to Graph
    # TODO: Do we need to transform as a Graph again?
    # It was a Graph before we transform into Dataset for training.
    G = to_networkx(data, to_undirected=True)

    # Define the edges weights as the values from the Attention matrix
    for i in range(len(weight)):
        G.add_edge(src[i].item(), tgt[i].item(), weight=weight[i].item())

    return G


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


def create_adj_matrix(edge_index):
    new_adj_matrix = np.zeros((GRAPH_NUMBER_NODES, GRAPH_NUMBER_NODES))

    src = edge_index[0].detach().numpy()
    tgt = edge_index[1].detach().numpy()

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

    data = create_demo_graph(GRAPH_NUMBER_NODES, GRAPH_NUMBER_CLASSES)

    adj_matrix = create_adj_matrix(data.edge_index)

    b_matrix = calc_t_order_neighbors(adj_matrix=adj_matrix, t=1)
    
    b_src_tgt_tensor = adj_to_edge_index(b_matrix)

    # After build the edge_index format, calculate the weights.
    b_weights = []
    
    for i in range(len(b_src_tgt_tensor[0])):
        b_weights.append(adj_matrix[b_src_tgt_tensor[0][i], b_src_tgt_tensor[1][i]])

    b_weights = torch.tensor(b_weights)
    
    b_edge_index = Data(edge_index=b_src_tgt_tensor, edge_attr=b_weights)

    data, att_tuple = gae_runner.run_training(
        epochs, data, b_edge_index, GRAPH_NUMBER_CLASSES
    )
    
    aggl_result = run_aggl_clustering(att_tuple)
    
    print(normalized_mutual_info_score(data.y.tolist(), aggl_result))

    return


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()

    epochs = args.epochs

    logger.info("Considering %s epochs", epochs)

    run(epochs)
