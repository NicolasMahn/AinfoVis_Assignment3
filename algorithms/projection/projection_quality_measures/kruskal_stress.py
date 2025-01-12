import numpy as np

from algorithms.distance_measures import distance_pipeline

def kruskal_stress_quality_measure(points, rd_points, params):
    d = distance_pipeline(params.get("distance_measure", "Euclidean"), points)
    rd_d = distance_pipeline(params.get("distance_measure", "Euclidean"), rd_points)

    return np.sqrt(np.sum(np.square(d - rd_d)) / np.sum(np.square(d)))