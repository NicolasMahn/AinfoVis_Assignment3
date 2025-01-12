import numpy as np

from ...distance_measures import distance_pipeline


def calculate_jaccard_coefficient(_, clusters, classes):
    if classes is None:
        raise ValueError("Classes must be provided to calculate the Jaccard Coefficient.")

    # Calculate pairwise distances in both cluster assignments
    clusters = np.array(clusters)
    classes = np.array(classes)
    cluster_matrix = distance_pipeline('Hamming', clusters.reshape(-1, 1), ) == 0
    class_matrix = distance_pipeline('Hamming', classes.reshape(-1, 1)) == 0

    # Compute True Positives, False Positives, and False Negatives
    TP = np.logical_and(cluster_matrix, class_matrix).sum()
    FP = np.logical_and(cluster_matrix, np.logical_not(class_matrix)).sum()
    FN = np.logical_and(np.logical_not(cluster_matrix), class_matrix).sum()

    # Calculate and return the Jaccard Coefficient
    return TP / (TP + FP + FN)