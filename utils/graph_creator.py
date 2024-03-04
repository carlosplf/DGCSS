import logging

from torch_geometric.datasets import RandomPartitionGraphDataset


def create_demo_graph():
    """
    Using the RandomPartitionGraphDataset from Torch,
    create a synthetic Graph.

    return: dataset 0 position data structure.
    """
    
    logger = logging.getLogger(__name__)

    logger.info("Creating a new demo graph using RandomPartitionGraphDataset...")

    data = RandomPartitionGraphDataset(
        root="",
        num_classes=5,
        num_nodes_per_class=100,
        node_homophily_ratio=0.5,
        average_degree=12,
        num_channels=32,
    )

    return data[0]
