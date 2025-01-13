from sklearn.manifold import MDS

from ..distance_measures import distance_pipeline

def mds_projection(points, params):
    """
    Perform MDS on a set of points using the scikit-learn implementation.
    :param points: a list of points in n-dimensional space
    :param max_iterations: maximum number of iterations to perform
    :param epsilon: a small value to prevent division by zero
    :return: the low-dimensional representation of the points and the history of the low-dimensional representation
    """
    mds = MDS(params.get("n_components", 2), random_state=params.get("random_state", 42), metric=True, n_init=1,
              max_iter=params.get("max_iterations", 1000), eps=params.get("epsilon", 1e-7))
    rd_points = mds.fit_transform(points)

    return rd_points