import argparse
import logging

from runners import gae_runner
from utils.graph_creator import get_cora_dataset
from utils.b_matrix import BMatrix


parser = argparse.ArgumentParser()

parser.add_argument(
    "--epochs", type=int, help="Define number of EPOCHS for training.", default=10
)
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument(
    "-fa",
    "--find_centroids_alg",
    type=str,
    help="Define the method to find \
        centroids. Options: KMeans, FastGreedy.",
    default="KMeans",
)
parser.add_argument(
    "--error_log_file",
    type=str,
    help="Define the CSV file name to \
        save error logs.",
    default="error_log.csv",
)
parser.add_argument(
    "-cl",
    "--c_loss_gama",
    type=int,
    help="Define the multiplier for Clustering Loss.",
    default=20,
)
parser.add_argument(
    "-pi",
    "--p_interval",
    type=int,
    help="Define the interval for calculating P.",
    default=10,
)

# TODO: Make this dynamic. Is set based on Cora dataset.
GRAPH_NUMBER_NODES = 250
GRAPH_NUMBER_CLASSES = 5


def run(epochs, find_centroids_alg, error_log_file, c_loss_gama, p_interval):
    data = get_cora_dataset()

    b_matrix = BMatrix(GRAPH_NUMBER_NODES)

    b_matrix.calc_t_order_neighbors(data, t=2)
    b_matrix.create_edge_index()

    runner = gae_runner.GaeRunner(
        epochs,
        data,
        b_matrix.edge_index,
        GRAPH_NUMBER_CLASSES,
        find_centroids_alg,
        c_loss_gama,
        p_interval,
    )

    runner.error_log_filename = error_log_file

    data, att_tuple = runner.run_training()

    return True


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    epochs = args.epochs
    c_loss_gama = args.c_loss_gama
    p_interval = args.p_interval
    find_centroids_alg = args.find_centroids_alg
    error_log_file = args.error_log_file

    logging.info("Considering %s epochs", epochs)

    run(epochs, find_centroids_alg, error_log_file, c_loss_gama, p_interval)
