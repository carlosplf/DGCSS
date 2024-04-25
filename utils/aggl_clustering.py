
import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering


def att_tuple_aggl_clustering(att_tuple, n_nodes):
    """
    Run the AgglomerativeClustering algorithm and try to classify nodes
    based on the Attention matrix.
    Distances of nodes are 1/<attention_coefficient>.
    Args:
        att_tuple: [[]] Attention values from GAE.
    Return: None
    """
    logging.debug("Running AgllomerativeClustering...")

    new_att = np.zeros((n_nodes, n_nodes))

    src = att_tuple[0][0].detach().numpy()
    tgt = att_tuple[0][1].detach().numpy()
    weight = att_tuple[1].detach().numpy()

    for i in range(len(src)):
        # Distances between nodes are 1/<attention_coefficient>
        new_att[src[i]][tgt[i]] = 1/weight[i]

    aggl_cl = AgglomerativeClustering(linkage="complete", n_clusters=5, metric='precomputed')
    aggl_result = aggl_cl.fit_predict(X=new_att)
    
    logging.debug("Finished running AgllomerativeClustering...")

    return aggl_result