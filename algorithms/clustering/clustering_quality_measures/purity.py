import numpy as np

def calculate_purity(_, cluster_labels, class_labels):
    # Find the unique clusters and classes
    clusters = np.unique(cluster_labels)
    class_labels = np.array(class_labels)

    # Initialize a counter for the correctly classified samples
    correct_classified = 0

    # Calculate purity by iterating over each cluster
    for cluster in clusters:
        # Select the samples belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_true_labels = class_labels[cluster_indices]

        # Find the most common true label in this cluster
        most_common_label = np.bincount(cluster_true_labels.astype(int)).argmax()

        # Count how many samples have this label in the current cluster
        correct_classified += np.sum(cluster_true_labels == most_common_label)

    # Calculate purity as the proportion of correctly classified samples
    purity = correct_classified / len(class_labels)
    return purity