from sklearn.cluster import DBSCAN


def sklearn_dbscan_clustering(data, params):
    model = DBSCAN(eps=params.get("epsilon", 0.5), min_samples=params.get("min_numbs", 5))
    return model.fit_predict(data)