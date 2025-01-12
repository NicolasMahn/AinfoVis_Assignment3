
from algorithms.clustering.k_means import kmeans_clustering
from algorithms.clustering.dbscan_sklearn import sklearn_dbscan_clustering
from algorithms.clustering.dbscan import dbscan_clustering
from algorithms.clustering.hierarchical_clustering import hierarchical_clustering


clustering_algorithms = {
    "K-Means": kmeans_clustering,
    "Sklearn DBSCAN": sklearn_dbscan_clustering,
    "DBSCAN": dbscan_clustering,
    "Hierarchical Clustering": hierarchical_clustering
}

def get_available_clustering_algorithms():
    return list(clustering_algorithms.keys())

def get_clustering_algorithm(algorithm_name):
    return clustering_algorithms[algorithm_name]

def clustering_pipeline(clustering_method, data, params=None):
    clustering_func = get_clustering_algorithm(clustering_method)
    return clustering_pipeline_using_func(clustering_func, data, params)

def clustering_pipeline_using_func(clustering_func, data, params=None):
    if params is None:
        params = {}
    return clustering_func(data, params)
