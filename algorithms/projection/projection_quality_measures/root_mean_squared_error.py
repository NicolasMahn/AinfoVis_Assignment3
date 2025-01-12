import numpy as np

from algorithms.distance_measures import distance_pipeline


def root_mean_squared_error_quality_measure(points, rd_points, params):
    y_true = distance_pipeline(params.get("distance_measure", "Euclidean"), points)
    y_pred = distance_pipeline(params.get("distance_measure", "Euclidean"), rd_points)

    return np.sqrt(np.mean(np.square(y_true - y_pred)))