import logging

from torch_geometric.datasets import RandomPartitionGraphDataset
from torch_geometric.datasets import Planetoid


def create_demo_graph(number_nodes, number_classes):
    """
    Using the RandomPartitionGraphDataset from Torch,
    create a synthetic Graph.

    return: dataset 0 position data structure.
    """
    logging.info("Creating a new demo graph using RandomPartitionGraphDataset...")

    data = RandomPartitionGraphDataset(
        root="",
        num_classes=number_classes,
        num_nodes_per_class=(round(number_nodes / number_classes)),
        node_homophily_ratio=0.5,
        average_degree=12,
        num_channels=64,
    )

    return data[0]


def get_planetoid_dataset(name="Cora"):
    """
    Get Cora Planetoid dataset.
    More info at https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
    """
    logging.info("Getting Planetoid dataset. Name: " + name)
    return Planetoid(root="", name=name)
