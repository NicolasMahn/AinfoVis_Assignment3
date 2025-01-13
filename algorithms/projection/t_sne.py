from sklearn.manifold import TSNE

from ..distance_measures import distance_pipeline

def t_sne_projection(data, params):
    model = TSNE(params.get("n_components", 2))
    return model.fit_transform(data)