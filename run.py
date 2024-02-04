import argparse
import random

import numpy as np
import torch
from torch_geometric.utils import to_networkx

from cora_dataset import planetoid_dataset
from data_loader.data_loader import load_as_graph
from runners import gae_runner
from utils.graph_viewer import plot_weights
from utils.utils import remove_edges

parser = argparse.ArgumentParser()
parser.add_argument("--planetoid", action="store_true", help="Use a Planetoid dataset.")
parser.add_argument(
    "--bench", action="store_true", help="Use a Benchmark clustering graph dataset."
)
parser.add_argument(
    "--epochs", type=int, help="Define number of EPOCHS for training.", default=10
)
parser.add_argument(
    "--n_graphs", type=int, help="Define number of Graphs to run.", default=1
)
parser.add_argument(
    "--level",
    type=str,
    help="Define the difificulty level from the Dataset. Options: 'easy_small', 'hard_small', 'easy', 'hard'.",
    default="easy_small",
)

# Defining random seeds
random.seed(81)
np.random.seed(81)
torch.manual_seed(81)
torch.cuda.manual_seed(81)


def save_plots(graph_index, G, features):
    # Plot original graph with edge weights
    filename = "before-" + str(graph_index) + ".png"
    plot_weights(G, features, folder_path="./charts", filename=filename)

    group_size_fraction = 5
    G = remove_edges(G, group_size_fraction)

    filename = "after-" + str(graph_index) + ".png"
    plot_weights(G, features, folder_path="./charts", filename=filename)


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


def run(epochs, dataset_to_use, n_graphs, level):
    data = None
    dataset = None

    # Defining PubMed as default!
    planetoid_dataset_name = "PubMed"

    if dataset_to_use == "planetoid":
        dataset = planetoid_dataset.download_dataset(planetoid_dataset_name)
        data = dataset[0]
        print("TODO: missing implementation fix for Plaetoid datasets... Exiting")
        return

    elif dataset_to_use == "bench":
        # graph_index sets the Graph to use inside the Dataset.
        # the easy_small configuration comes with 300 graphs to use.

        # Testing with some Graphs in sequence
        graph_index = 0
        for i in range(n_graphs):
            print("==> Testing with Graph ID", graph_index)
            graph, features = load_as_graph(dataset_name=level, graph_index=graph_index)
            data, att_tuple = gae_runner.run_training(epochs, graph)
            G = set_attention_as_weights(data, att_tuple)
            save_plots(i, G, features)
            graph_index += 1
        return

    else:
        print("No dataset specified. Exiting...")


if __name__ == "__main__":
    dataset_to_use = "bench"
    gae_bool = True
    args = parser.parse_args()

    if args.planetoid:
        print("Using Planetoid datasets...")
        dataset_to_use = "planetoid"

    elif args.bench:
        print("Using Benchmark Graphs...")
        dataset_to_use = "bench"

    epochs = args.epochs
    n_graphs = args.n_graphs
    level = args.level

    print("Considering", epochs, "epochs...")

    run(epochs, dataset_to_use, n_graphs, level)
