
from algorithms.clustering.clustering_quality_measures.jaccard_coefficient import calculate_jaccard_coefficient
from algorithms.clustering.clustering_quality_measures.silhouette_score import calculate_silhouette_score
from algorithms.clustering.clustering_quality_measures.davies_bouldin_index import calculate_davies_bouldin_index
from algorithms.clustering.clustering_quality_measures.purity import calculate_purity

clustering_quality_measures = {
    "Jaccard Coefficient": calculate_jaccard_coefficient,
    "Silhouette Score": calculate_silhouette_score,
    "Davies Bouldin Index": calculate_davies_bouldin_index,
    "Purity": calculate_purity
}

def get_available_clustering_quality_measures():
    return list(clustering_quality_measures.keys())

def get_clustering_quality_measure(quality_measure):
    return clustering_quality_measures[quality_measure]

def clustering_quality_measure_pipeline(quality_measure_method, data, clusters, classes=None):
    clustering_func = get_clustering_quality_measure(quality_measure_method)
    return clustering_quality_measure_pipeline_using_func(clustering_func, data, clusters, classes)

def clustering_quality_measure_pipeline_using_func(quality_measure_func, data, clusters, classes=None):
    return quality_measure_func(data, clusters, classes)
