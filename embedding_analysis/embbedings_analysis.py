import ast
import math
import pickle
import re
import shutil

import numpy as np
import os

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from tqdm import tqdm

from algorithms.clustering import clustering_pipeline
from algorithms.projection import projection_pipeline
from .embedding_function import get_embedding_function
from .populate_database import DatabaseManager
from util import load_config

PINK = "\033[38;5;205m"
RESET = "\033[0m"
ORANGE = "\033[38;5;208m"
WHITE = "\033[97m"
BLUE = "\033[34m"

class EmbeddingsAnalysis:
    def __init__(self, selected_topics :list=None, word_for_word :bool=False, update :bool=False,
                 debug :bool=False, batch_size :int=100):
        if selected_topics is None:
            selected_topics = []
        self.selected_topics = selected_topics
        self.word_for_word = word_for_word
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
            self.topic_configs = [data_topics[config['default_topic']]]
        else:
            self.topic_configs = [data_topics[selected_topic] for selected_topic in self.selected_topics]

        self.embeddings = []
        self.topic_labels = []
        self.id_labels = []
        self.word_labels = []

        for topic in self.topic_configs:
            topic_embeddings = []
            t_topic_labels = []
            t_id_labels = []
            t_word_labels = []
            if not update:
                t_topic_labels, t_id_labels, t_word_labels, topic_embeddings = (
                    self.load_embeddings_and_labels(topic['topic_dir']))
                if self.debug:
                    print(f"{ORANGE}🔍  Data Loaded{RESET}")
            if len(topic_embeddings) == 0:
                if self.debug:
                    print(f"{ORANGE}⌛  Embeddings need to be calculated, this will take time.{RESET}")
                self.populater = DatabaseManager(topic["topic_dir"], separate_in_chunks=not self.word_for_word,
                                                 chunk_separation_warning=True)
                t_topic_labels, t_id_labels, t_word_labels, topic_embeddings = self.embedding_process(topic)
                if self.debug:
                    print(f"{ORANGE}✅  Embeddings calculated and saved{RESET}")
            self.embeddings.extend(topic_embeddings)
            self.topic_labels.extend(t_topic_labels)
            self.id_labels.extend(t_id_labels)
            self.word_labels.extend(t_word_labels)

    def get_similarity_matrix(self, recalculate_similarity=False):
        file_name = f"similarity_matrix_{str([tc['topic_name'] for tc in self.topic_configs])}"

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
        file_name = f"reduced_similarities_{algorithm}_{str([tc['topic_name'] for tc in self.topic_configs])}"

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
        file_name = f"{algorithm}_clusters_{str([tc['topic_name'] for tc in self.topic_configs])}"

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

    def get_id_labels(self):
        return self.id_labels

    def get_word_labels(self):
        return self.word_labels

    def get_embeddings(self):
        return self.embedding

    def embedding_process(self, topic):
        if self.update:
            self.delete_embeddings_if_no_checkpoint(topic['topic_dir'])

        topic_labels, id_labels, word_labels, documents = self.get_topic_batch_documents(topic)

        bar_format = f"{WHITE}⌛  Embedding {topic['topic_name']} {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"
        with tqdm(total=(self.num_batches - self.completed_batches), bar_format=bar_format, unit="batch", ) as pbar:
            first_iteration = True
            while not self.done:
                if not first_iteration:
                    topic_labels, id_labels, word_labels, documents = self.get_topic_batch_documents(topic)
                first_iteration = False

                if len(documents) == 0:
                    break

                batch_embeddings = self.embbed_documents(documents)
                self.save_extend_labels(topic_labels, id_labels, word_labels, topic['topic_dir'])
                self.save_extend_embeddings(batch_embeddings, topic['topic_dir'])
                self.final_save_checkpoint(topic['topic_dir'])
                pbar.update(1)
                self.completed_batches += 1


        return self.load_embeddings_and_labels(topic['topic_dir'])

    def get_topic_batch_documents(self, topic):
        word_labels = []
        topic_labels = []
        id_labels = []
        documents = []
        topic_dir = topic['topic_dir']
        document_dir = f"{topic_dir}/documents"

        file_names, file_num, chunk_num, word_num = self.load_checkpoint(topic_dir)
        first_iteration = False
        if len(file_names) == 0:

            file_names = os.listdir(document_dir)
            first_iteration = True
            self.num_batches = math.ceil(len(file_names)/self.batch_size) #assuming no chunks, no word_for_word
            self.completed_batches = 0
            self.save_checkpoint(topic_dir, file_names, 0, 0, 0)
            self.done = False

        while file_num < len(file_names):
            file = file_names[file_num]
            if not file.endswith(".txt"):
                continue

            d_documents = self.populater.process_txt(f"{document_dir}/{file}")
            while chunk_num < len(d_documents):
                d_document = d_documents[chunk_num]
                document = d_document.page_content
                id_ = d_document.id

                if self.word_for_word:
                    words = document.split()
                    if first_iteration:
                        self.num_batches = math.ceil((len(file_names)*len(words))/self.batch_size)  #assuming word count is similar
                        first_iteration = False
                    # remove punctuation
                    words = [word.strip(".,!?") for word in words]
                    # remove duplicates
                    words = list(set(words))

                    while word_num < len(words):
                        word = words[word_num]
                        documents.append(word)
                        word_labels.append(word)
                        id_labels.append(id_)
                        topic_labels.append(topic['topic_name'])
                        word_num += 1

                        if len(documents) >= self.batch_size:
                            break
                    if not len(documents) >= self.batch_size:
                        chunk_num += 1
                        word_num = 0

                else:
                    documents.append(document)
                    id_labels.append(id_)
                    topic_labels.append(topic['topic_name'])
                    chunk_num += 1

                if len(documents) >= self.batch_size:
                    break

            if len(documents) >= self.batch_size:
                self.pre_save_checkpoint(file_names, file_num, chunk_num, word_num)
                return topic_labels, id_labels, word_labels, documents
            else:
                chunk_num = 0
                file_num += 1

        self.delete_checkpoint(topic_dir)
        self.done = True
        return topic_labels, id_labels, word_labels, documents

    def embbed_documents(self, documents):
        embedding_function = get_embedding_function()
        embeddings = np.array(embedding_function.embed_documents(documents))
        return embeddings

    def calculate_cosine_similarity(self):
        # Calculate the cosine similarity matrix
        similarity_matrix = cosine_similarity(self.embeddings)
        return similarity_matrix

    def run_projection_pipline(self, algorithm, params):
        reduced_similarities = projection_pipeline(algorithm, self.get_similarity_matrix(), params)
        return reduced_similarities

    def run_clustering_pipline(self, algorithm, params):
        clusters = clustering_pipeline(algorithm, self.get_similarity_matrix(), params)
        return clusters

    def load_labels(self, topic_dir):
        try:
            if self.word_for_word:
                topic_labels = np.load(f"{topic_dir}/embeddings/word_topic_labels.npy")
                id_labels = np.load(f"{topic_dir}/embeddings/word_id_labels.npy")
                word_labels = np.load(f"{topic_dir}/embeddings/word_word_labels.npy")
            else:
                topic_labels = np.load(f"{topic_dir}/embeddings/topic_labels.npy")
                id_labels = np.load(f"{topic_dir}/embeddings/id_labels.npy")
                word_labels = []
            return topic_labels, id_labels, word_labels
        except FileNotFoundError:
            return [], [], []

    def save_labels(self, topic_labels, id_labels, word_labels, topic_dir):
        os.makedirs(f"{topic_dir}/embeddings", exist_ok=True)
        if self.word_for_word:
            np.save(f"{topic_dir}/embeddings/word_topic_labels", topic_labels)
            np.save(f"{topic_dir}/embeddings/word_id_labels", id_labels)
            np.save(f"{topic_dir}/embeddings/word_word_labels", word_labels)
        else:
            np.save(f"{topic_dir}/embeddings/topic_labels", topic_labels)
            np.save(f"{topic_dir}/embeddings/id_labels", id_labels)
            np.save(f"{topic_dir}/embeddings/word_labels", word_labels)

    def save_extend_labels(self, topic_labels, id_labels, word_labels, topic_dir):
        try:
            old_topic_labels, old_id_labels, old_word_labels = self.load_labels(topic_dir)
            topic_labels = np.concatenate((old_topic_labels, topic_labels))
            id_labels = np.concatenate((old_id_labels, id_labels))
            word_labels = np.concatenate((old_word_labels, word_labels))
        except FileNotFoundError:
            pass
        self.save_labels(topic_labels, id_labels, word_labels, topic_dir)

    def load_embeddings_and_labels(self, topic_dir):
        try:
            if self.word_for_word:
                topic_labels = np.load(f"{topic_dir}/embeddings/word_topic_labels.npy")
                id_labels = np.load(f"{topic_dir}/embeddings/word_id_labels.npy")
                word_labels = np.load(f"{topic_dir}/embeddings/word_word_labels.npy")
                embeddings = np.load(f"{topic_dir}/embeddings/word_embeddings.npy")
            else:
                topic_labels = np.load(f"{topic_dir}/embeddings/topic_labels.npy")
                id_labels = np.load(f"{topic_dir}/embeddings/id_labels.npy")
                embeddings = np.load(f"{topic_dir}/embeddings/embeddings.npy")
                word_labels = []
            return topic_labels, id_labels, word_labels, embeddings
        except FileNotFoundError:
            return [], [], [], []

    def save_embeddings_and_labels(self, embeddings, topic_labels, id_labels, word_labels, topic_dir):
        os.makedirs(f"{topic_dir}/embeddings", exist_ok=True)
        if self.word_for_word:
            np.save(f"{topic_dir}/embeddings/word_topic_labels", topic_labels)
            np.save(f"{topic_dir}/embeddings/word_id_labels", id_labels)
            np.save(f"{topic_dir}/embeddings/word_word_labels", word_labels)
            np.save(f"{topic_dir}/embeddings/word_embeddings", embeddings)
        else:
            np.save(f"{topic_dir}/embeddings/topic_labels", topic_labels)
            np.save(f"{topic_dir}/embeddings/id_labels", id_labels)
            np.save(f"{topic_dir}/embeddings/embeddings", embeddings)

    def save_extend_embeddings(self, embeddings, topic_dir):
        os.makedirs(f"{topic_dir}/embeddings", exist_ok=True)
        if self.word_for_word:
            file_name = f"{topic_dir}/embeddings/word_embeddings.npy"
        else:
            file_name = f"{topic_dir}/embeddings/embeddings.npy"
        try:
            existing_embeddings = np.load(file_name)
            embeddings = np.concatenate((existing_embeddings, embeddings))
        except FileNotFoundError:
            pass
        np.save(file_name, embeddings)

    def load_checkpoint(self, topic_dir):
        try:
            with open(f"{topic_dir}/embeddings/checkpoint.txt", "r") as file:
                file_num, chunk_num, word_num, num_batches, completed_batches = file.read().split()
                self.num_batches = int(num_batches)
                self.completed_batches = int(completed_batches)
            file_names = np.load(f"{topic_dir}/embeddings/file_names.npy")
            return file_names, int(file_num), int(chunk_num), int(word_num)
        except FileNotFoundError:
            return [], 0, 0, 0

    def pre_save_checkpoint(self, file_names, file_num, chunk_num, word_num):
        self.pre_save_checkpoint_data = f"{file_num} {chunk_num} {word_num} {self.num_batches} {self.completed_batches}"
        self.pre_save_file_names_data = file_names

    def final_save_checkpoint(self, topic_dir):
        os.makedirs(f"{topic_dir}/embeddings", exist_ok=True)
        with open(f"{topic_dir}/embeddings/checkpoint.txt", "w") as file:
            file.write(self.pre_save_checkpoint_data)
        np.save(f"{topic_dir}/embeddings/file_names", self.pre_save_file_names_data)

    def save_checkpoint(self, topic_dir, file_names, file_num, chunk_num, word_num):
        os.makedirs(f"{topic_dir}/embeddings", exist_ok=True)
        with open(f"{topic_dir}/embeddings/checkpoint.txt", "w") as file:
            file.write(f"{file_num} {chunk_num} {word_num} {self.num_batches} {self.completed_batches}")
        np.save(f"{topic_dir}/embeddings/file_names", file_names)

    def delete_checkpoint(self, topic_dir):
        os.remove(f"{topic_dir}/embeddings/checkpoint.txt")
        os.remove(f"{topic_dir}/embeddings/file_names.npy")

    def delete_embeddings_if_no_checkpoint(self, topic_dir):
        os.makedirs(f"{topic_dir}/embeddings", exist_ok=True)
        if not os.path.exists(f"{topic_dir}/embeddings/checkpoint.txt"):
            print("deleting")
            shutil.rmtree(f"{topic_dir}/embeddings")

    def save_precalculated_data(self, data, filename):
        os.makedirs(f"data/precalculated_data", exist_ok=True)
        topic_names = "_".join([topic['topic_name'].lower() for topic in self.topic_configs])
        with open(f"data/precalculated_data/{filename}_{topic_names}.pkl", 'wb') as f:
            pickle.dump(data, f)

    def load_precalculated_data(self, filename):
        try:
            topic_names = "_".join([topic['topic_name'].lower() for topic in self.topic_configs])
            with open(f"data/precalculated_data/{filename}_{topic_names}.pkl", 'rb') as f:
                data = pickle.load(f)
            return data
        except (FileNotFoundError, pickle.UnpicklingError):
            return []