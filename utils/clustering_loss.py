import numpy as np
import logging
from sklearn.cluster import KMeans


def get_clusters_centroids(Z, n_clusters):
    """
    Runs KMeans clustering to find the centroids.
    """
    logging.info("Calculating centroids with K-Means...")
    X = Z.cpu().detach().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    return kmeans.cluster_centers_


def update_clusters_centers(clusters_centroids, Q_grad, step_size=0.01):
    """
    Update the cluster centroids based on the current clustering loss.
    Args:
        [[]]: clusters centroids.
        Q_grad: gradient array from Q. 
        (float) step_size: multiplier for the gradient subtraction
    Return:
        (array): New clusters centroids array.
    """
    logging.info("Updating clusters centroids...")
    new_centroids = []

    for u in range(len(clusters_centroids)):
        tmp_array = clusters_centroids[u] - (step_size * np.mean(Q_grad[u].detach().numpy()))
        new_centroids.append(tmp_array)
    
    return new_centroids


def calculate_q(clusters_centroids, Z):
    """
    Calculate Q using Student’s t-distribution.
    """

    if clusters_centroids is None:
        return None
    
    nodes = Z.cpu().detach().numpy()
    number_of_nodes = len(nodes)
    number_of_centroids = len(clusters_centroids)

    Q = np.zeros((number_of_nodes, number_of_centroids))

    for i in range(number_of_nodes):
        for u in range(number_of_centroids):
            q = 1 / ( 1 + np.power(np.linalg.norm(nodes[i]-clusters_centroids[u]), 2) )
            q_lower_sum = 0
            for k in range(number_of_centroids):
                q_lower_sum += 1 / ( 1 + np.power(np.linalg.norm(nodes[i]-clusters_centroids[k]), 2) )
            q = q / q_lower_sum
            Q[i][u] = q        

    return Q


def calculate_p(Q):
    """
    Calculate P using Q.
    """
    
    if Q is None:
        return None

    logging.info("Calculating P...")
    
    number_of_nodes = len(Q)
    number_of_centroids = len(Q[0])

    P = np.zeros((number_of_nodes, number_of_centroids))

    for i in range(number_of_nodes):
        for u in range(number_of_centroids):
            p = np.power(Q[i][u], 2) / sum(Q[x][u] for x in range(number_of_nodes))
            p_lower_sum = 0
            for k in range(number_of_centroids):
                p_lower_sum += (np.power(Q[i][k], 2) / sum(Q[x][k] for x in range(number_of_nodes)))
            p = p / p_lower_sum
            P[i][u] = p        
    return P


def calculate_clustering_loss(Q, P):

    if P is None or Q is None:
        return 1000
    
    number_of_nodes = len(Q)
    number_of_centroids = len(Q[0])
    
    loss_clustering = 0

    for i in range(number_of_nodes):
        for u in range(number_of_centroids):
            loss_clustering += (P[i][u] * (np.log(P[i][u]/Q[i][u])))

    return loss_clustering
