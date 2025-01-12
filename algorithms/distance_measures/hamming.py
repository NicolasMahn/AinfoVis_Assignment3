import numpy as np

def calculate_hamming_distance(points):
    """
    Calculate the pairwise Hamming distances between points in n-dimensional space
    following the formula: D_{ij} = \sum_{k=1}^{n} \delta(x_{ik}, x_{jk})
    :param points: a list of points in n-dimensional space
    :return: a matrix of pairwise Hamming distances between points
    """
    points = np.array(points)
    distances = np.sum(points[:, np.newaxis] != points[np.newaxis, :], axis=-1)
    return distances