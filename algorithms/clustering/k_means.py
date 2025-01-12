from sklearn.cluster import KMeans

def kmeans_clustering(data, params):
    model = KMeans(n_clusters=params.get("n_clusters", 3))
    return model.fit_predict(data)