from algorithms.projection.projection_quality_measures.kruskal_stress import kruskal_stress_quality_measure
from algorithms.projection.projection_quality_measures.root_mean_squared_error import \
    root_mean_squared_error_quality_measure
from algorithms.projection.projection_quality_measures.sammon_stress import sammon_stress_quality_measure

projection_quality_measures = {
    "RMSE": root_mean_squared_error_quality_measure,
    "Kruskal Stress": kruskal_stress_quality_measure,
    "Sammon Stress": sammon_stress_quality_measure
}

def get_available_projection_quality_measures():
    return list(projection_quality_measures.keys())

def get_projection_quality_measure(quality_measure):
    return projection_quality_measures[quality_measure]

def projection_quality_measure_pipeline(quality_measure_method, points, rd_points, params=None):
    projection_func = get_projection_quality_measure(quality_measure_method)
    return projection_quality_measure_pipeline_using_func(projection_func, points, rd_points, params)

def projection_quality_measure_pipeline_using_func(quality_measure_func, points, rd_points, params=None):
    if params is None:
        params = {}
    return quality_measure_func(points, rd_points, params)
