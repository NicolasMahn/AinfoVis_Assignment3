from sklearn.metrics import silhouette_score

def calculate_silhouette_score(data, clusters, _):
    if len(set(clusters)) > 1:
        return silhouette_score(data, clusters)
    else:
        return None