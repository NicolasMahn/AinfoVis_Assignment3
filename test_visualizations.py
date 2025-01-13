from embedding_analysis.article_analysis import ArticleAnalysis
import pandas as pd
import math
import plotly.express as px
import plotly.subplots as sp
import numpy as np
from tqdm import tqdm

from algorithms.clustering import get_available_clustering_algorithms
from algorithms.distance_measures import distance_pipeline
from algorithms.projection import get_available_projection_algorithms

NO_SAMMON_MAPPING = True  # Sammon mapping is disabled by default as it simply takes to long to compute
NO_HOME_BREW_DBSCAN = True


WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"


def comparing_articles(aa):
    df = pd.DataFrame({
        'x': None,
        'y': None,
        'Cluster': None,
        'Topic': aa.get_topic_labels()
    })

    projection_grid(aa, df, "Topic", "Test Plot")

def projection_grid(ea, df, color, plot_name, class_preservation=False, k=5):
    # Retrieve the available projection algorithms
    available_projection_algorithms = get_available_projection_algorithms()
    if NO_SAMMON_MAPPING and 'Sammon Mapping' in available_projection_algorithms:
        available_projection_algorithms.remove('Sammon Mapping')

    num_projection_algorithms = len(available_projection_algorithms)

    # Calculate grid dimensions
    grid_size = math.ceil(math.sqrt(num_projection_algorithms))

    # Initialize plotly subplots with calculated grid size
    fig = sp.make_subplots(rows=grid_size, cols=grid_size, subplot_titles=available_projection_algorithms)

    if color in get_available_clustering_algorithms():
        clusters_int = ea.get_clusters(color)
        clusters = []
        for d in clusters_int:
            if d == -1:
                clusters.append('Noise')
            else:
                clusters.append(str(d))

        df["Cluster"] = clusters

    if color == 'Review Type':
        color_discrete_map = {'positive': 'blue', 'negative': 'red'}
    else:
        color_discrete_map = None

    if color in get_available_clustering_algorithms():
        color = 'Cluster'

    hover_data = [col for col in df.columns if col not in ['x', 'y'] and df[col].notna().any()]

    # Iterate over each algorithm and create a subplot
    if class_preservation:
        bar_format = (f"{WHITE}⌛  Plotting Projection Methods with Class Preservation...   "
                      f"{{l_bar}}{PINK}{{bar}}{WHITE}{{r_bar}}{RESET}")
    else:
        bar_format = f"{WHITE}⌛  Plotting Projection Methods...   {{l_bar}}{ORANGE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=num_projection_algorithms, bar_format=bar_format) as pbar:
        for idx, algorithm in enumerate(available_projection_algorithms):

            # Generate the reduced points for each algorithm
            points = ea.get_reduced_similarities(algorithm=algorithm)
            df['x'], df['y'] = points[:, 0], points[:, 1]

            if class_preservation:
                # Compute class preservation for each point
                class_preservation = []
                for i, point in df.iterrows():
                    distances = np.linalg.norm(points - point[['x', 'y']].values.astype(float), axis=1)
                    nearest_indices = np.argsort(distances)[1:k + 1]
                    same_class_count = sum(df.iloc[nearest_indices][color] == point[color])
                    class_preservation.append(same_class_count / k)
                df['Class Preservation'] = class_preservation

                scatter = px.scatter(df, x='x', y='y', color='Class Preservation', hover_data=hover_data,
                                     color_continuous_scale='Cividis')
            else:
                scatter = px.scatter(df, x='x', y='y', color=color, hover_data=hover_data,
                                     color_discrete_map=color_discrete_map)

            # Determine subplot position
            row = (idx // grid_size) + 1
            col = (idx % grid_size) + 1

            # Add the scatter plot to the subplot
            for trace in scatter.data:
                fig.add_trace(trace, row=row, col=col)
            pbar.update(1)

    # Update layout and axis visibility
    fig.update_layout(height=900, width=900, title_text=plot_name)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()

def main():
    aa = ArticleAnalysis(["ars-technica"], update=False, reset=True)


    # comparing_articles(aa)


if __name__ == "__main__":
    main()