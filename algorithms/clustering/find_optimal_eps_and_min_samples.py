from copyreg import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt
from sympy.stats.sampling.sample_numpy import numpy
import pickle

from algorithms.clustering.clustering_pipeline import clustering_algorithms, clustering_pipeline
from embedding_analysis.review_analysis import ReviewAnalysis
from util import open_csv_file


def main():
    test_set = "blobs"
    ideal_clusters = 2  # Adjust as needed
    tolerance = 0  # Adjust as needed

    # Example usage
    eps_values = np.arange(20.0, 40.0, 0.1)  # Adjust range and step as needed
    min_samples_values = range(3, 20)  # Adjust as needed

    # data = open_csv_file(f"../../data/test_clustering_projection/{test_set}.csv")
    with open(f"../../data/precalculated_data/similarity_matrix_['Reviews']_reviews.pkl", 'rb') as f:
        data = pickle.load(f)

    best_params = grid_search_dbscan_params(data, eps_values, min_samples_values)
    find_eps_using_k_distances(data, best_params['min_samples'], test_set)
    print(f"Best params: eps = {best_params['eps']}, min_samples = {best_params['min_samples']}")

    print(f"With the known number of clusters, we can find the parameters resulting in {ideal_clusters} clusters:")
    params_resulting_in_ideal_clusters = find_dbscan_params_for_cluster_count(data, ideal_clusters, eps_values,
                                                                              min_samples_values, tolerance)
    for result in params_resulting_in_ideal_clusters:
        print(result)


def find_dbscan_params_for_cluster_count(data, target_clusters, eps_values, min_samples_values, tolerance=1):
    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = clustering_pipeline("DBSCAN", data, {"epsilon": eps, "min_numbs": min_samples})

            # Count the number of clusters (ignoring noise points, labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Calculate how close the result is to the target number of clusters
            cluster_diff = abs(target_clusters - n_clusters)

            # Check if the number of clusters is within the desired tolerance range
            if cluster_diff <= tolerance:
                # Calculate Silhouette Score if there is more than one cluster
                if n_clusters > 1:
                    score = silhouette_score(data, labels)
                else:
                    score = -1  # Use -1 as a placeholder if there's only one cluster

                # Store parameters, cluster count, and quality score
                results.append({'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters,
                                'silhouette_score': score, 'cluster_diff': cluster_diff})

    # Sort results by silhouette score (descending) and then by cluster_diff (ascending) for best matches
    results = sorted(results, key=lambda x: (-x['silhouette_score'], x['cluster_diff']))

    return results

def find_eps_using_k_distances(data, min_samples, test_set=None):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, min_samples - 1])  # Sort distances to the k-th nearest neighbor
    plt.plot(distances)
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    plt.xlabel('Data Points (sorted)')
    if test_set:
        plt.title(f'Elbow Method for Epsilon on {test_set} Dataset')
    else:
        plt.title('Elbow Method for Epsilon')
    plt.show()


def grid_search_dbscan_params(data, eps_values, min_samples_values):
    best_params = {'eps': None, 'min_samples': None, 'score': -1}
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            # Only calculate score if more than 1 cluster
            if len(set(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_params['score']:
                    best_params = {'eps': eps, 'min_samples': min_samples, 'score': score}
    return best_params


if __name__ == "__main__":
    main()