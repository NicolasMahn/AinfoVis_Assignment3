import time
import numpy as np
from tqdm import tqdm
from sklearn.manifold import MDS

from ..distance_measures import distance_pipeline

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"

def sammon_projection(data, params):
    learning_rate = params.get("learning_rate", 0.1)
    max_iterations = params.get("max_iterations", 1000)
    threshold = params.get("threshold", 1e-7)
    epsilon = params.get("epsilon", 1e-7)
    orthogonal = params.get("orthogonal", False)
    progress_bar = params.get("progress_bar", False)
    distance_measure = params.get("distance_measure", "Euclidean")
    return sammon_mapping(data, learning_rate, max_iterations, threshold, epsilon, orthogonal, distance_measure,
                          progress_bar)


def sammon_mapping(points, learning_rate, max_iterations, threshold, epsilon, orthogonal, distance_measure,
                   progress_bar=True):
    """
    Perform Sammon mapping on a set of points.
    :param points: a list of points in n-dimensional space
    :param learning_rate: step size for updating the low-dimensional representation of the points
    :param max_iterations: maximum number of iterations to perform
    :param threshold: a small change in the pairwise distances in the low-dimensional space to stop early
    :param epsilon: a small value to prevent division by zero
    :param orthogonal: initialize the low-dimensional representation of the points using the top two dimensions
    :param animate: if the history of the low-dimensional representation of the points should be saved
    :return: the low-dimensional representation of the points
    """
    if progress_bar:
        bar_format = f"{WHITE}⌛  Analyzing Points  {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"
        with tqdm(total=max_iterations, bar_format=bar_format, unit="iteration") as pbar:
            return _sammon_mapping(points, learning_rate, max_iterations, threshold, epsilon, orthogonal,
                                   distance_measure, pbar)
    else:
        return _sammon_mapping(points, learning_rate, max_iterations, threshold, epsilon, orthogonal,
                               distance_measure, None)


def _sammon_mapping(points, learning_rate, max_iterations, threshold, epsilon, orthogonal, distance_measure, pbar):
    points = np.array(points)
    previous_rd_distances = None

    # Step 1: Initialize the low-dimensional representation of the points
    rd_points = _generate_rd_points(points, orthogonal)

    # Step 2: Calculate the pairwise distances in the high-dimensional space
    distances = distance_pipeline(distance_measure, points)

    for i in range(max_iterations):
        # Step 3: Calculate the pairwise distances in the low-dimensional space
        rd_distances = distance_pipeline(distance_measure, rd_points)

        # Check if the change in rd_distances is below the threshold
        if previous_rd_distances is not None:
            change = np.linalg.norm(rd_distances - previous_rd_distances)
            if change < threshold:
                if pbar is not None:
                    pbar.update(max_iterations - i)
                    time.sleep(0.5)
                    print(f"Stopping early at iteration {i} due to small change in distances: {change}")
                break
        previous_rd_distances = rd_distances

        # Step 4: Update the low-dimensional representation of the points
        rd_points = _update_points(rd_points, rd_distances, distances, learning_rate, epsilon)

        if pbar is not None:
            pbar.update(1)

    return rd_points


def _update_points(rd_points, rd_distances, distances, learning_rate, epsilon):
    """
    Update the low-dimensional representation of the points using the Sammon mapping algorithm.
    Using these algorithms:
     p_i <- p_i - \alpha \Delta_i
     \Delta_i = \frac{\frac{\partial E}{\partial p_i}}{\left|\frac{\partial^2 E}{\partial p_i^2}\right|}
     \frac{\partial E}{\partial p_i} = - \frac{2}{\sum_{i < j} D_{ij}} \sum_{j \neq i} (\frac{D_{ij} -
                                       d_{ij}}{d_{ij} D_{ij}})(p_i - p_j)
     \frac{\partial^2 E}{\partial p_i^2} = - \frac{2}{\sum_{i < j} D_{ij}} \sum_{j \neq i} \frac{1}{D_[ij} d_{ij}} *
                                           ((D_{ij} - d_{ij}) - \frac{(p_i - p_j)^2}{d_{ij}} *
                                                                (1 + \frac{D_{ij} - d_{ij}}[d_{ij}}))
    :param rd_points: low-dimensional representation of the points
    :param rd_distances: low-dimensional pairwise distances between points
    :param distances: distance_measures matrix of the high-dimensional points
    :param learning_rate: step size for updating the low-dimensional representation of the points
    :param epsilon: a small value to prevent division by zero
    :return: updated low-dimensional representation of the points
    """

    n = len(rd_points)

    d_minus_rdd = distances - rd_distances
    d_times_rdd = distances * rd_distances + epsilon
    rdd_plus_e = rd_distances + epsilon

    for i in range(n):
        i_minus_j = rd_points[i] - rd_points

        first_derivative =  ( # - 2/c *
                            np.sum([(d_minus_rdd[i, j] / d_times_rdd[i, j]) *
                                    (i_minus_j[j]) for j in range(n) if j != i], axis=0))
        first_derivative = first_derivative * -1 # since we are not multiplying by -2/c
                                                 # (and taking the absolut of the 2nd derivative)

        second_derivative = ( # - 2/c *
                            np.sum([1 / d_times_rdd[i, j] *
                                    ((d_minus_rdd[i, j]) - (i_minus_j[j])**2 / rdd_plus_e[i, j] *
                                     (1 + d_minus_rdd[i, j] / rdd_plus_e[i, j]))
                                    for j in range(n) if j != i], axis=0))
        second_derivative = np.abs(second_derivative) + epsilon

        gradient = first_derivative / second_derivative

        rd_points[i] -= learning_rate * gradient

    return rd_points


def _generate_random_rd_points(n, dim=2):
    return np.random.rand(n, dim)


def _generate_orthogonal_rd_points(points, dim=2):
    variances = np.var(points, axis=0)
    top_two_indices = np.argsort(variances)[-dim:]
    rd_points = points[:, top_two_indices]
    return rd_points


def _generate_rd_points(points, orthogonal):
    if orthogonal:
        return _generate_orthogonal_rd_points(points)
    else:
        return _generate_random_rd_points(len(points))