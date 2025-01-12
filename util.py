import csv
import os
import yaml

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"

def load_config(config_file=None):
    if config_file is None:
        config_file = 'config.yaml'

    base_path = os.path.dirname(__file__)  # Get current file directory
    config_file = os.path.join(base_path, config_file)

    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def open_csv_file(data_source):
    base_path = os.path.dirname(__file__)  # Get current file directory
    data_source = os.path.join(base_path, data_source)

    points = list()

    # Open and read the CSV file
    if os.path.exists(data_source):
        with open(data_source, mode='r', newline='') as file:
            reader = csv.reader(file)

            # Read each row in the CSV file
            for index, point in enumerate(reader):
                coordinates = list()
                for coor in point:
                    if isinstance(coor, str):
                        try:
                            coordinates.append(float(coor))
                        except Exception as e:
                            print(f"{RED}An error occurred while trying to convert a coordinate of a point from str to float: "
                                  f"{e}{RESET}")
                    else:
                        coordinates.append(coor)
                points.append(tuple(coordinates))

        return points
    else:
        print(f"{RED}Error: File '{data_source}' does not exist. {RESET}")
        os._exit(1)


def open_csv_labels(label_source):
    base_path = os.path.dirname(__file__)  # Get current file directory
    label_source = os.path.join(base_path, label_source)


    points = open_csv_file(label_source)
    # Assuming each point in the labels file is a single label
    labels = [point[0] for point in points]
    return labels


def open_edgelist_file(data_source):
    base_path = os.path.dirname(__file__)  # Get current file directory
    data_source = os.path.join(base_path, data_source)

    with open(data_source, 'r') as file:
        lines = file.readlines()
        graph = {}
        labels = {}
        current_label = 0
        for line in lines:
            node1, node2 = line.strip().split()
            if node1 not in labels:
                labels[node1] = current_label
                current_label += 1
            if node2 not in labels:
                labels[node2] = current_label
                current_label += 1
            node1 = labels[node1]
            node2 = labels[node2]
            if node1 not in graph:
                graph[node1] = []
            if node2 not in graph:
                graph[node2] = []
            graph[node1].append(node2)
            graph[node2].append(node1)
    return graph, labels


def load_clustering_projection_labels(name):
    # Function to load actual labels for each dataset
    return open_csv_labels(f"data/test_clustering_projection/{name}_labels.csv")


def load_clustering_projection_data(name):
    # Function to load data for each dataset
    return open_csv_file(f"data/test_clustering_projection/{name}.csv")


def load_test_network(name):
    return open_edgelist_file(f"data/test_networks/{name}.edgelist")