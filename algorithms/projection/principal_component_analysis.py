from sklearn.decomposition import PCA

from ..distance_measures import distance_pipeline

def pca_projection(data, params):
    """
    Perform Principal Component Analysis (PCA) on a set of points.
    :param data: a list of points in n-dimensional space
    :param params: a dictionary of parameters
    :return: the low-dimensional representation of the points
    """

    pca = PCA(n_components=params.get("n_components", 2), random_state=params.get("random_state", 42))
    return pca.fit_transform(data)