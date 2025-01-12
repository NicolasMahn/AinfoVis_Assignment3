import numpy as np

def calulate_cosine_distance(points):
    """
    Calculate the cosine distance between points in n-dimensional space
    following the formula: D_{ij} = 1 - \frac{x_i \cdot x_j}{||x_i|| \cdot ||x_j||}
    :param points: a list of points in n-dimensional space
    :return: a matrix of cosine distances between points
    """
    points = np.array(points)
    dot_product = np.dot(points, points.T)
    norm = np.linalg.norm(points, axis=1)
    norm_product = np.outer(norm, norm)
    distances = 1 - dot_product / norm_product
    return distances