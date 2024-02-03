import torch
import numpy as np
import random
import argparse
from cora_dataset import planetoid_dataset
from runners import gae_runner
from data_loader.data_loader import load_as_graph


parser = argparse.ArgumentParser()
parser.add_argument("--planetoid", action="store_true",
                    help="Use a Planetoid dataset.")
parser.add_argument("--bench", action="store_true",
                    help="Use a Benchmark clustering graph dataset.")
parser.add_argument("--epochs", type=int,
                    help="Define number of EPOCHS for training.",
                    default=10)


# Defining random seeds
random.seed(81)
np.random.seed(81)
torch.manual_seed(81)
torch.cuda.manual_seed(81)


def run(epochs, dataset_to_use):

    data = None
    dataset = None

    # Defining PubMed as default!
    planetoid_dataset_name = "PubMed"

    if dataset_to_use == "planetoid":
        dataset = planetoid_dataset.download_dataset(planetoid_dataset_name)
        data = dataset[0]
        print(data.x)
        print(type(data.x))
        print(len(data.x))
        print(len(data.x[0]))
        return

    elif dataset_to_use == "bench":
        # graph_index sets the Graph to use inside the Dataset.
        # the easy_small configuration comes with 300 graphs to use.
        graph, features = load_as_graph(graph_index=0)
        gae_runner.run_training(epochs, graph, features)
        return

    else:
        print("No dataset specified. Exiting...")

    gae_runner.run_training(epochs, data)


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

    print("Considering", epochs, "epochs...")

    run(epochs, dataset_to_use)
