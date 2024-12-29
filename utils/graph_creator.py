import logging

from torch_geometric.datasets import RandomPartitionGraphDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Twitch


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


def get_dataset(name="Cora", ds_type="Planetoid"):
    if ds_type == "Planetoid":
        return get_planetoid_dataset(name=name)

    if ds_type == "Twitch":
        dataset_name = "DE"
        return get_twitch_dataset(name=dataset_name)


def get_planetoid_dataset(name="Cora"):
    """
    Get Cora Planetoid dataset.
    More info at https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
    """
    logging.info("Getting Planetoid dataset. Name: " + name)
    return Planetoid(root="", name=name)


def get_twitch_dataset(name="DE"):
    """
    Get Torch Twitch dataset.
    """
    logging.info("Getting Twitch dataset. Name: " + name)
    return Twitch(root="", name=name)
