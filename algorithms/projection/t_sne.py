from sklearn.manifold import TSNE

from ..distance_measures import distance_pipeline

def t_sne_projection(data, params):

    distance_matrix = distance_pipeline(params.get("distance_measure", "Euclidean"), data)

    model = TSNE(params.get("n_components", 2), random_state=params.get("random_state", 42), metric='precomputed',
                 init='random')
    return model.fit_transform(distance_matrix)