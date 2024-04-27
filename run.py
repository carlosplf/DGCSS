import argparse
import logging

from runners import gae_runner
from utils.graph_creator import get_cora_dataset
from utils.b_matrix import BMatrix


parser = argparse.ArgumentParser()

parser.add_argument(
    "--epochs", type=int, help="Define number of EPOCHS for training.", default=10
)


GRAPH_NUMBER_NODES = 250
GRAPH_NUMBER_CLASSES = 5


def run(epochs):

    data = get_cora_dataset()

    b_matrix = BMatrix(GRAPH_NUMBER_NODES)

    b_matrix.calc_t_order_neighbors(data, t=2)
    b_matrix.create_edge_index()

    runner = gae_runner.GaeRunner(epochs, data, b_matrix.edge_index, GRAPH_NUMBER_CLASSES)

    data, att_tuple = runner.run_training()

    return


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    epochs = args.epochs

    logging.info("Considering %s epochs", epochs)

    run(epochs)
