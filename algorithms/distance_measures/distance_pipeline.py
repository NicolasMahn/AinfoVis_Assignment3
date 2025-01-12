
from .euclidean import calculate_euclidean_distance
from .cosine import calulate_cosine_distance
from .hamming import calculate_hamming_distance
from .jaccard import calculate_jaccard_distance


distance_algorithms = {
    "Euclidean": calculate_euclidean_distance,
    "Cosine": calulate_cosine_distance,
    "Hamming": calculate_hamming_distance,
    "Jaccard": calculate_jaccard_distance
}

def get_available_distance_algorithms():
    return list(distance_algorithms.keys())

def get_distance_algorithm(algorithm_name):
    return distance_algorithms[algorithm_name]

def distance_between_two_points_pipeline(distance_method, point1, point2):
    matrix = [point1, point2]
    distamce_matrix = distance_pipeline(distance_method, matrix)
    return distamce_matrix[0][1]

def distance_pipeline(distance_method, data):
    distance_func = get_distance_algorithm(distance_method)
    return distance_pipeline_using_func(distance_func, data)

def distance_pipeline_using_func(distance_func, data):
    return distance_func(data)