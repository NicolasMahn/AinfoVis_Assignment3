# from langchain_community.embeddings.ollama import OllamaEmbeddings

import openai
from scrt import OPENAI_KEY

"""
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
"""

class EmbeddingFunction:
    def __init__(self):
        openai.api_key = OPENAI_KEY

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, query):
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding

def get_embedding_function():
    return EmbeddingFunction()
