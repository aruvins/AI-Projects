from sentence_transformers import SentenceTransformer


"""
This module provides a function to embed text using the SentenceTransformer model.
The SentenceTransformer model is a pre-trained model that converts sentences into
dense vector representations (embeddings). These embeddings can be used for various 
natural language processing tasks such as semantic search, clustering, and classification.

The `embed_text` function takes a string input and returns its corresponding embedding vector.
"""
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text)