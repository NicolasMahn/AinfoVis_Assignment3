import pandas as pd
import math
import plotly.express as px
import plotly.subplots as sp
import numpy as np

from algorithms.clustering import get_available_clustering_algorithms
from algorithms.distance_measures import distance_pipeline
from algorithms.projection import get_available_projection_algorithms

from embedding_analysis.embbedings_analysis import EmbeddingsAnalysis
from embedding_analysis.article_analysis import ArticleAnalysis
import matplotlib.colors as mcolors
import plotly.colors as pc

from numpy import unique
WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"


def normalize_to_range(values, new_min=0.0, new_max=1.0):
    min_val = min(values)
    max_val = max(values)
    return [new_min + (x - min_val) / (max_val - min_val) * (new_max - new_min) for x in values]


def interpolate_color(value, color1, color2):
    """
    Interpolates a HEX color based on a value between -1 and 1.
    Args:
        value (float): A number between -1 and 1.
        color1 (str): The starting HEX color (e.g., "#FFFF00").
        color2 (str): The ending HEX color (e.g., "#0000FF").
    Returns:
        str: The interpolated HEX color.
    """
    # Special case: if value is -1, return grey
    if value == -1:
        return "#808080"

    # Ensure value is within bounds (0 to 1)
    value = max(0, min(1, value))

    # Convert HEX to RGB tuples
    color1_rgb = tuple(int(color1[i:i + 2], 16) for i in (1, 3, 5))
    color2_rgb = tuple(int(color2[i:i + 2], 16) for i in (1, 3, 5))

    # Interpolate each RGB component
    interpolated_rgb = tuple(
        int(color1_rgb[i] + (color2_rgb[i] - color1_rgb[i]) * value)
        for i in range(3)
    )

    # Convert RGB tuple back to HEX
    return "#{:02X}{:02X}{:02X}".format(*interpolated_rgb)


def map_values_to_colors(values, color1, color2):
    """
    Maps a list of values between -1 and 1 to HEX colors.
    Args:
        values (list of float): A list of numbers between -1 and 1.
        color1 (str): The starting HEX color.
        color2 (str): The ending HEX color.
    Returns:
        list of str: A list of interpolated HEX colors.
    """
    return [interpolate_color(value, color1, color2) for value in values]

def get_color_map(df, selected_filter):
    if selected_filter == "Published Dates":

        filtered_dates = unique([x for x in df[selected_filter] if x is not None])


        # Convert dates to numerical format
        dates = pd.to_datetime(filtered_dates)
        date_nums = [date.timestamp() for date in dates]
        normalized_dates= normalize_to_range(date_nums)

        color_map = {"None": "#808080"}
        color_list = map_values_to_colors(normalized_dates, "#0000FF", "#FFFF00")
        for i, date in enumerate(filtered_dates):

            color_map[pd.to_datetime(date).date()] = color_list[i]

    else:
        unique_values = df[selected_filter].unique()
        color_map = {value: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, value in enumerate(unique_values)}
    return color_map


def get_graphs(selected_topics, selected_filter, projection_algorithm, query, color_query_value, result):
    aa = ArticleAnalysis(selected_topics, update=False)

    # Get similarity scores to the query
    similarities = aa.get_similarity_to_query(query)

    df = pd.DataFrame({
        'x': None,
        'y': None,
        'Cluster': None,
        'Author': aa.get_authors(),
        'Title': aa.get_titles(),
        'URL': aa.get_urls(),
        'PublishedAt': aa.get_published_datetimes(),
        'Published Dates': aa.get_published_dates(),
        'Source': aa.get_source_names(),
        'Source URL': aa.get_source_urls(),
        'Source Category': aa.get_source_categories(),
        'Similarity to Query': similarities
    })

    scatter_fig = projection_graph(aa, df, selected_filter, projection_algorithm, color_query_value)
    similarity_fig = similarity_matrix(aa, df, selected_filter)
    similarity_graph_fig = similarity_graph(aa, df, query, selected_filter)
    result['scatter'] = scatter_fig
    result['similarity'] = similarity_fig
    result['similarity_graph'] = similarity_graph_fig


def projection_graph(ea, df, color, projection_algorithm="T-SNE", color_query_value=False):
    """
    :param ea:
    :param df:
    :param color:
    :param projection_algorithm: Options: "PCA", "MDS", "T-SNE"
    :return:
    """

    if color not in df.columns:
        color = "Source"
    if color_query_value:
        color = "Similarity to Query"


    hover_data = [col for col in df.columns if col not in ['x', 'y'] and df[col].notna().any()]

    # Generate the reduced points for each algorithm
    points = ea.get_reduced_similarities(algorithm=projection_algorithm)
    df['x'], df['y'] = points[:, 0], points[:, 1]

    color_map = get_color_map(df, color)
    scatter = px.scatter(df, x='x', y='y', color=color, hover_data=hover_data, color_discrete_map=color_map,
                         title=f'{projection_algorithm} Projection of Articles')
    scatter.update_layout(title_x=0.5, xaxis=dict(visible=False), yaxis=dict(visible=False), height=800)
    return scatter


def similarity_matrix(ea, df, selected_filter):
    """
    Generate a similarity matrix and plot it, either for groups defined by a filter or for all embeddings.

    :param ea: An object with the method `get_similarity_matrix()` that returns the full similarity matrix.
    :param df: A pandas DataFrame containing metadata about embeddings, including the selected filter.
    :param selected_filter: A column name in `df` to group embeddings (e.g., authors).
    :return: A plotly figure visualizing the similarity matrix.
    """

    if selected_filter not in df.columns or selected_filter == "Published Dates":
        selected_filter = "Source"

    # Generate the full similarity matrix
    full_similarity_matrix = ea.get_similarity_matrix()

    if selected_filter in df.columns:
        # Get unique groups in the filter
        unique_groups = df[selected_filter].unique()
        group_count = len(unique_groups)

        # Initialize an empty similarity matrix for groups
        similarity_matrix = np.zeros((group_count, group_count))

        # Map each group to its indices in the DataFrame
        group_indices = {group: df[df[selected_filter] == group].index for group in unique_groups}

        # Calculate pairwise similarities between groups
        for i, group_i in enumerate(unique_groups):
            indices_i = group_indices[group_i]
            for j, group_j in enumerate(unique_groups):
                indices_j = group_indices[group_j]

                # Extract pairwise similarities from the full similarity matrix
                similarities = full_similarity_matrix[np.ix_(indices_i, indices_j)]

                # Compute the mean similarity between the two groups
                similarity_matrix[i, j] = np.mean(similarities)

    else:
        # If no filter is provided, use the full similarity matrix
        similarity_matrix = full_similarity_matrix

    # Generate the similarity matrix plot
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x=selected_filter, y=selected_filter, color="Similarity"),
        x=unique_groups if selected_filter in df.columns else None,
        y=unique_groups if selected_filter in df.columns else None,
        title='Similarity Matrix between ' + selected_filter,
    )

    # Update layout to hide the x-axis
    fig.update_layout(xaxis=dict(visible=False), title_x=0.5)

    return fig

def similarity_graph(ea, df, query, selected_filter):
    """
    Generate a similarity graph comparing each item in the selected topics to the query, split by selected filter groups.

    :param ea: An object with the method `get_similarity_to_query(query)` that returns the similarity scores.
    :param df: A pandas DataFrame containing metadata about embeddings.
    :param query: The query text to compare against.
    :param selected_filter: The filter to group the data by.
    :return: A plotly figure visualizing the similarity graph.
    """


    # Create a beeswarm plot
    color_map = get_color_map(df, selected_filter)
    beeswarm_fig = px.strip(df, x='Similarity to Query', y=selected_filter, orientation='h', stripmode='overlay',
                            title='Similarity of ' + selected_filter + ' to "' + query + '"',
                            color=selected_filter, color_discrete_map=color_map)

    # Update layout to make the figure more oval-shaped
    beeswarm_fig.update_layout(
        height=600,
        width=1200,
        title_x=0.5
    )

    return beeswarm_fig