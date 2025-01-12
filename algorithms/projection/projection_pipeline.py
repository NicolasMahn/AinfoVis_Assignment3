from algorithms.projection.t_sne import t_sne_projection
from algorithms.projection.sammon_mapping import sammon_projection
from algorithms.projection.principal_component_analysis import pca_projection
from algorithms.projection.multidimensional_scaling import mds_projection


projection_algorithms = {
    "PCA": pca_projection,
    "MDS": mds_projection,
    "Sammon Mapping": sammon_projection,
    "T-SNE": t_sne_projection
}

def get_available_projection_algorithms():
    return list(projection_algorithms.keys())

def get_projection_algorithm(algorithm_name):
    return projection_algorithms[algorithm_name]

def projection_pipeline(projection_method, data, params=None):
    projection_func = get_projection_algorithm(projection_method)
    return projection_pipeline_using_func(projection_func, data, params)

def projection_pipeline_using_func(projection_func, data, params=None):
    if params is None:
        params = {}
    return projection_func(data, params)