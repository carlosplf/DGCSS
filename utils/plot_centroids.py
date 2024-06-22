from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def plot_centroids(all_nodes, centroids):
    """
    Apply PCA (2) to all nodes and centroids and plot.
    Args:
        (list of list): all_nodes.
        (list of list): centroids (also nodes).
    """
    pca = PCA(n_components=2)   
    all_nodes_pca = pca.fit_transform(all_nodes.detach().numpy())

    pca = PCA(n_components=2)
    centroids_pca = pca.fit_transform(centroids)

    print(all_nodes_pca)
    print("===========")
    print(centroids_pca)

    all_nodes_x = [row[0] for row in all_nodes_pca]
    all_nodes_y = [row[1] for row in all_nodes_pca]

    centroids_x = [row[0] for row in centroids_pca]
    centroids_y = [row[1] for row in centroids_pca]
    
    _plot(all_nodes_x, all_nodes_y, centroids_x, centroids_y)

    return None


def _plot(all_nodes_x, all_nodes_y, centroids_x, centroids_y):
    plt.scatter(all_nodes_x, all_nodes_y, alpha=0.5)
    plt.scatter(centroids_x, centroids_y, c=['#FF0000'], alpha=0.5)
    plt.savefig('test.png')