import numpy as np

def calculate_jaccard_distance(points):
    """
    Calculate the pairwise Jaccard distances between points in n-dimensional space
    following the formula: D_{ij} = 1 - \frac{|A_i \cap A_j|}{|A_i \cup A_j|}
    :param points: a list of points in n-dimensional space
    :return: a matrix of pairwise Jaccard distances between points
    """
    points = np.array(points)
    intersection = np.sum(points[:, np.newaxis] == points[np.newaxis, :], axis=-1)
    union = np.sum(points[:, np.newaxis] | points[np.newaxis, :], axis=-1)
    distances = 1 - intersection / union
    return distances