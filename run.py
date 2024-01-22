import torch
import numpy as np
import random
import argparse
from utils.graph_creator import define_graph
from utils.graph_creator import create_from_dataset
from cora_dataset import planetoid_dataset
from runners import gat_runner
from runners import gae_runner


parser = argparse.ArgumentParser()
parser.add_argument("--dummy", action="store_true",
                    help="Use a dummy smal graph for testing.")
parser.add_argument("--cora", action="store_true",
                    help="Use the Cora Planetoid dataset.")
parser.add_argument("--gae", action="store_true",
                    help="Use GAE encoder and decoder.")
parser.add_argument("--epochs", type=int,
                    help="Define number of EPOCHS for training.",
                    default=100)


# Defining random seeds
random.seed(81)
np.random.seed(81)
torch.manual_seed(81)
torch.cuda.manual_seed(81)


def run(epochs, dataset_to_use, gae_bool=False):

    data = None
    dataset = None

    if dataset_to_use == "cora":
        dataset = planetoid_dataset.download_dataset()
        data = dataset[0]
        G, communities = create_from_dataset(data)
        X = data.x
    elif dataset_to_use == "dummy":
        G, communities = define_graph()
        # Adding some features to the Graph
        X = torch.tensor(np.eye(18), dtype=torch.float)
    else:
        print("No dataset specified. Exiting...")

    if not gae_bool:
        # If gae_bool is false, we should run GAT directly
        gat_runner.run_training(dataset=dataset, epochs=epochs)
        return

    else:
        # TODO: move this to the method that builds the Graph
        G.features = X

        for i in range(len(G.nodes())):
            G.nodes[i]['features'] = X[i]
            G.nodes[i]['label'] = communities[i]

        gae_runner.run_training(epochs, G, communities, dataset_to_use)


if __name__ == "__main__":
    dataset_to_use = "cora"
    gae_bool = False
    args = parser.parse_args()

    if args.cora:
        print("Using Cora Planetoid dataset...")
        dataset_to_use = "cora"

    elif args.dummy:
        print("Using Dummy dataset...")
        dataset_to_use = "dummy"

    if args.gae:
        print("GAE set to TRUE. Using Encoder and Decoder...")
        gae_bool = True

    epochs = args.epochs

    print("Considering", epochs, "epochs...")

    run(epochs, dataset_to_use, gae_bool)
