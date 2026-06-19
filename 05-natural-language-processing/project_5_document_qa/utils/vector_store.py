from utils.embeddings import embed_text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = []

    def add_document(self, chunks):
        """
        Adds a document to the vector store by chunking it and embedding each chunk.

        Args:
            document (str): The input document to be added to the vector store.
        """
        self.chunks = chunks
        
        self.embeddings = [
            embed_text(chunk) for chunk in chunks
        ]

    def search(self, query, top_k = 3):
        query_embedding = embed_text(query)
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        top_indicies = np.argsort(similarities)[-top_k:][::-1]

        return [self.chunks[i] for i in top_indicies]