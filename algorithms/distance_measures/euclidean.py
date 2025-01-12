import numpy as np

def calculate_euclidean_distance(points):
    """
    Calculate the pairwise distances between points in n-dimensional space
    following the formula: D_{ij} = \sqrt{\sum_{k=1}^{n} (x_{ik} - x_{jk})^2}
    :param points: a list of points in n-dimensional space
    :return: a matrix of pairwise distances between points
    """
    points = np.array(points)
    distances = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :])**2, axis=-1))
    return distances