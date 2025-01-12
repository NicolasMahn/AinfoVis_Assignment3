from sklearn.metrics import silhouette_score, davies_bouldin_score

def calculate_davies_bouldin_index(data, clusters, _):
    if len(set(clusters)) > 1:
        return davies_bouldin_score(data, clusters)
    else:
        return None