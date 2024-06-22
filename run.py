import argparse
import logging

from runners import gae_runner
from utils.graph_creator import get_cora_dataset
from utils.b_matrix import BMatrix


parser = argparse.ArgumentParser()

parser.add_argument(
    "--epochs", type=int, help="Define number of EPOCHS for training.",
    default=10
)
parser.add_argument(
    '-d', '--debug', action='store_true'
)
parser.add_argument(
    "--find_centroids_alg", type=str, help="Define the method to find \
        centroids. Options: KMeans, FastGreedy.", default='KMeans'
)
parser.add_argument(
    "--error_log_file", type=str, help="Define the CSV file name to save error logs.", default='error_log.csv'
)


GRAPH_NUMBER_NODES = 250
GRAPH_NUMBER_CLASSES = 5


def run(epochs, find_centroids_alg, error_log_file):

    data = get_cora_dataset()

    b_matrix = BMatrix(GRAPH_NUMBER_NODES)

    b_matrix.calc_t_order_neighbors(data, t=2)
    b_matrix.create_edge_index()

    runner = gae_runner.GaeRunner(
        epochs,
        data,
        b_matrix.edge_index,
        GRAPH_NUMBER_CLASSES,
        find_centroids_alg
    )

    runner.error_log_filename = error_log_file

    data, att_tuple = runner.run_training()

    return


if __name__ == "__main__":

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    epochs = args.epochs
    find_centroids_alg = args.find_centroids_alg
    error_log_file = args.error_log_file

    logging.info("Considering %s epochs", epochs)

    run(epochs, find_centroids_alg, error_log_file)
