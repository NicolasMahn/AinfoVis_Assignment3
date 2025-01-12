from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering(data, params):
    model = AgglomerativeClustering(
        n_clusters=params.get("n_clusters", 3),
        metric=params.get("affinity", "euclidean"),
        linkage=params.get("linkage", "ward"),
    )
    return model.fit_predict(data)
