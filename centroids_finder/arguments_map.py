from centroids_finder import (
    random_seeds,
    fastgreedy,
    kcore,
    weighted_modularity,
    pagerank,
    kmeans,
    betweenness_centrality,
    weighted_betweenness_centrality,
    eigenvector_centrality,
    closeness_centrality,
    weighted_closeness_centrality,
)


map = {
    "Random": random_seeds,
    "BC": betweenness_centrality,
    "WBC": weighted_betweenness_centrality,
    "PageRank": pagerank,
    "KMeans": kmeans,
    "FastGreedy": fastgreedy,
    "KCore": kcore,
    "EigenV": eigenvector_centrality,
    "CC": closeness_centrality,
    "WCC": weighted_closeness_centrality,
}
