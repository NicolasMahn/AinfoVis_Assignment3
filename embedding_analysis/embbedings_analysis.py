import ast
import math
import pickle
import re
import shutil
import hashlib
import numpy as np
import os

import pandas as pd
from langchain_community.vectorstores import Chroma
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from tqdm import tqdm

from algorithms.clustering import clustering_pipeline
from algorithms.projection import projection_pipeline

from .embedding_function import get_embedding_function, EmbeddingFunction
from .populate_database import DatabaseManager, populate_db
from util import load_config

PINK = "\033[38;5;205m"
RESET = "\033[0m"
ORANGE = "\033[38;5;208m"
WHITE = "\033[97m"
BLUE = "\033[34m"

class EmbeddingsAnalysis:
    def __init__(self, selected_topics :list=None, update :bool=True, reset :bool=False,
                 debug :bool=False, batch_size :int=100):
        if selected_topics is None:
            selected_topics = []
        self.selected_topics = selected_topics
        self.update = update
        self.debug = debug
        self.batch_size = batch_size

        self.num_batches = 0
        self.completed_batches = 0
        self.done = False
        self.pre_save_checkpoint_data = ""
        self.pre_save_file_names_data = []

        config = load_config()
        data_topics = config['data_topics']
        if len(selected_topics) == 0:
            default_topic = config['default_topic']
            data_topics[default_topic].update({"id": default_topic})
            self.topic_configs = [data_topics[default_topic]]
        elif "all" in self.selected_topics:
            self.topic_configs = []
            for topic in data_topics.keys():
                data_topics[topic].update({"id": topic})
                self.topic_configs.append(data_topics[topic])
        else:
            self.topic_configs = []
            for selected_topic in self.selected_topics:
                data_topics[selected_topic].update({"id": selected_topic})
                self.topic_configs.append(data_topics[selected_topic])

        self.embeddings = []
        self.topic_labels = []
        self.metadata = []
        self.ids = []

        for topic in self.topic_configs:
            if update or reset:
                populate_db(topic["id"], debug=debug, reset=reset)
            topic_embeddings, topic_metadata, topic_ids = self.get_embedding_from_chroma()
            self.embeddings.extend(topic_embeddings)
            self.topic_labels.extend([topic["topic_name"] for _ in range(len(topic_metadata))])
            self.metadata.extend(topic_metadata)
            self.ids.extend(topic_ids)


    def get_embedding_from_chroma(self):
        embeddings = []
        metadata = []
        ids = []
        for topic in self.topic_configs:
            chroma_dir = f"{topic['topic_dir']}/chroma"
            chroma = Chroma(persist_directory=chroma_dir, embedding_function=get_embedding_function())
            data = chroma.get(include=["embeddings", "metadatas"])
            embeddings.extend(data["embeddings"])
            metadata.extend(data["metadatas"])
            ids.extend(data["ids"])
        return embeddings, metadata, ids

    def get_similarity_matrix(self, recalculate_similarity=False):
        file_name = f"similarity_matrix"

        similarity_matrix = []
        if not recalculate_similarity:
            similarity_matrix = self.load_precalculated_data(file_name)
        if len(similarity_matrix) == 0 or len(similarity_matrix) != len(self.embeddings):
            similarity_matrix = self.calculate_cosine_similarity()
            self.save_precalculated_data(similarity_matrix, file_name)
            if self.debug:
                print(f"{ORANGE}🔗  Cosine Similarity calculated{RESET}")
        else:
            if self.debug:
                print(f"{ORANGE}🔍  Cosine Similarity loaded{RESET}")
        return similarity_matrix

    def get_reduced_similarities(self, algorithm, params=None, recalculate_projection=False):
        file_name = f"reduced_similarities"

        if params is None:
            params = {}
        dimensions = params.get("n_components", 2)
        reduced_similarities = []
        if not recalculate_projection:
            reduced_similarities = self.load_precalculated_data(file_name)
        if len(reduced_similarities) == 0 or len(reduced_similarities) != len(self.embeddings):
            reduced_similarities = self.run_projection_pipline(algorithm, params)
            self.save_precalculated_data(reduced_similarities, file_name)
            if self.debug:
                print(f"{ORANGE}➗  {dimensions}D {algorithm} calculated{RESET}")
        else:
            if self.debug:
                print(f"{ORANGE}🔍  {dimensions}D {algorithm} loaded{RESET}")
        return reduced_similarities

    def get_clusters(self, algorithm, params=None, recalculate_clusters=False):
        file_name = f"{algorithm}_clusters"

        clustering_labels = []
        if not recalculate_clusters:
            clustering_labels = self.load_precalculated_data(file_name)
        if len(clustering_labels) == 0 or len(clustering_labels) != len(self.embeddings):
            clustering_labels = self.run_clustering_pipline(algorithm, params)
            self.save_precalculated_data(clustering_labels,
                                         file_name)
            if self.debug:
                print(f"{ORANGE}🎨  Clusters calculated{RESET}")
        else:
            if self.debug:
                print(f"{ORANGE}🔍  Clusters loaded{RESET}")
        return clustering_labels

    def get_topic_labels(self):
        return self.topic_labels

    def get_metadata(self):
        return self.metadata

    def calculate_cosine_similarity(self):
        # Flatten the list of lists into a 2D array
        embeddings_2d = np.vstack(self.embeddings)
        similarity_matrix = self._get_cosine_similarity(embeddings_2d)
        return similarity_matrix

    def _get_cosine_similarity(self, embeddings):
        # Calculate the cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def run_projection_pipline(self, algorithm, params):
        reduced_similarities = projection_pipeline(algorithm, self.get_similarity_matrix(), params)
        return reduced_similarities

    def run_clustering_pipline(self, algorithm, params):
        clusters = clustering_pipeline(algorithm, self.get_similarity_matrix(), params)
        return clusters

    def save_precalculated_data(self, data, filename):
        os.makedirs(f"data/precalculated_data", exist_ok=True)
        topic_names = "_".join([topic['topic_name'].lower() for topic in self.topic_configs])
        hash_object = hashlib.md5(topic_names.encode())
        hash_topic_names = hash_object.hexdigest()
        with open(f"data/precalculated_data/{filename}_{hash_topic_names}.pkl", 'wb') as f:
            pickle.dump(data, f)

    def load_precalculated_data(self, filename):
        try:
            topic_names = "_".join([topic['topic_name'].lower() for topic in self.topic_configs])
            hash_object = hashlib.md5(topic_names.encode())
            hash_topic_names = hash_object.hexdigest()
            with open(f"data/precalculated_data/{filename}_{hash_topic_names}.pkl", 'rb') as f:
                data = pickle.load(f)
            return data
        except (FileNotFoundError, pickle.UnpicklingError):
            return []

    def get_similarity_to_query(self, query):
        # Implement the logic to compute similarity scores to the query
        # This is a placeholder implementation
        embedder = EmbeddingFunction()
        query_embedding = embedder.embed_query(query)
        return [self._get_cosine_similarity([query_embedding, embedding])[0][1] for embedding in self.embeddings]