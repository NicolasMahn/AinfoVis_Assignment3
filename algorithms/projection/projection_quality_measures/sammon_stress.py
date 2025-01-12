import numpy as np

from algorithms.distance_measures import distance_pipeline

def sammon_stress_quality_measure(points, rd_points, params):
    d = distance_pipeline(params.get("distance_measure", "Euclidean"), points)
    rd_d = distance_pipeline(params.get("distance_measure", "Euclidean"), rd_points)

    epsilon = params.get("epsilon", 1e-9)
    weighted_diff = (np.square(d - rd_d) / (d + epsilon))

    # Calculate Sammon stress
    return np.sum(weighted_diff) / np.sum(d)
