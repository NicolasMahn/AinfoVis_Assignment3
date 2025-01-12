from tqdm import tqdm

from ..distance_measures import distance_between_two_points_pipeline

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"

def dbscan_clustering(data, params):
    #TODO: add affinity
    return dbscan(data, params.get("epsilon", 0.5), params.get("min_numbs", 5), params.get("epsilon_lt", False),
                  params.get("min_numb_gt", False), params.get("self_neighbour", False),
                  params.get("progress_bar", False))



def _range_query(points, p1, epsilon, epsilon_lt):
    neighbours = list()

    for p2 in points:
        distance = distance_between_two_points_pipeline("Euclidean", p1, p2)
        if epsilon_lt:
            if abs(distance) < epsilon:
                neighbours.append(p2)
        else:
            if abs(distance) <= epsilon:
                neighbours.append(p2)
    return neighbours


def dbscan(np_points, epsilon, min_numb, epsilon_lt=False, min_numb_gt=False, self_neighbour=False, progress_bar=True):
    """
    Perform DBSCAN clustering on a set of points.
    :param points: a list of points in n-dimensional space
    :param epsilon: maximum distance_measures between two points to be considered neighbours
    :param min_numb: minimum number of points required to form a cluster
    :param epsilon_lt: if True, use < instead of <= for epsilon
    :param min_numb_gt: if True, use > instead of >= for min_numb
    :param self_neighbour: should a point be considered its own neighbour
    :return: a list of clusters and a list of noise points. The clusters are represented as lists of points.
    """

    if self_neighbour:
        min_numb -= 1

    points = []
    for n_point in np_points:
        points.append(tuple(n_point))

    if progress_bar:
        bar_format = f"{WHITE}⌛  Calculating Neighbours    {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"
        with tqdm(total=len(points), bar_format=bar_format, unit="point") as pbar:
            clusters, noise = _dbscan_process(points, epsilon, min_numb, epsilon_lt, min_numb_gt, pbar, progress_bar)

    else:
        pbar = None
        clusters, noise = _dbscan_process(points, epsilon, min_numb, epsilon_lt, min_numb_gt, pbar, progress_bar)

    labels = []
    for point in points:
        for i, cluster in enumerate(clusters):
            if point in cluster:
                labels.append(i)
                break
        if point in noise:
            labels.append(-1)

    return labels


def _dbscan_process(points, epsilon, min_numb, epsilon_lt, min_numb_gt, pbar, progress_bar):
    clusters = list()
    noise = list()
    labeled_points = set()

    for point in points:
        if point in labeled_points:
            continue
        labeled_points.add(point)

        neighbours = _range_query(points, point, epsilon, epsilon_lt)
        if (min_numb_gt and len(neighbours) < min_numb) or (not min_numb_gt and len(neighbours) <= min_numb):
            noise.append(point)
            if progress_bar:
                pbar.update(1)
            continue
        cluster = [point]

        seed = neighbours
        while seed:
            point = seed.pop()
            if point in labeled_points:
                continue
            labeled_points.add(point)

            neighbours = _range_query(points, point, epsilon, epsilon_lt)
            if (not min_numb_gt and len(neighbours) >= min_numb) or (min_numb_gt and len(neighbours) > min_numb):
                new_neighbours = [n for n in neighbours if n not in labeled_points]
                seed.extend(new_neighbours)
            cluster.append(point)
            if progress_bar:
                pbar.update(1)

        clusters.append(cluster)
        if progress_bar:
            pbar.update(1)
    return clusters, noise