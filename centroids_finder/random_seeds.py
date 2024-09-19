import logging
import random


def select_centroids(Z, num_of_communities=5):
    logging.info("Using Random Seeds to find the centroids...")

    centroids = random.sample(range(1, len(Z)), num_of_communities)

    clusters_centroids = []

    # Get Z values for each centroid.
    for c in centroids:
        clusters_centroids.append(Z[c].tolist())

    return clusters_centroids
