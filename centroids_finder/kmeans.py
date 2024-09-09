import logging
from sklearn.cluster import KMeans


def select_centroids(Z, n_clusters):
    """
    Runs KMeans clustering to find the centroids.
    """
    logging.info("Using K-Means to find centroids.")
    X = Z.cpu().detach().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    return kmeans.cluster_centers_
