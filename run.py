import argparse
import random
import logging

import numpy as np
import torch
from torch_geometric.utils import to_networkx

from runners import gae_runner
from utils.graph_viewer import plot_weights
from utils.utils import remove_edges
from utils.graph_creator import create_demo_graph


parser = argparse.ArgumentParser()

parser.add_argument(
    "--epochs", type=int, help="Define number of EPOCHS for training.", default=10
)

# Defining random seeds
random.seed(81)
np.random.seed(81)
torch.manual_seed(81)
torch.cuda.manual_seed(81)


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


def run(epochs):

    data = create_demo_graph()
    data, att_tuple = gae_runner.run_training(epochs, data)
    
    G_before = set_attention_as_weights(data, att_tuple)
    
    group_size_fraction = 1.50
    G_after = remove_edges(G_before, group_size_fraction)
    
    save_plots(0, G_before, G_after, data.y.tolist())
    return


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()

    epochs = args.epochs

    logger.info("Considering %s epochs", epochs)

    run(epochs)
