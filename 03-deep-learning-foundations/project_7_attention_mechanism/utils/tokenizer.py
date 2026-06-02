import torch

def tokenize(sentence):
    return sentence.lower().split()

def create_embeddings(tokens, embedding_dim=8):
    # For simplicity, we will create random embeddings for each token
    # In a real-world scenario, you would use pre-trained embeddings or train your own embeddings on a large corpus of text.
    # 
    # Returns: embeddings -> [num_tokens, embedding_dim]
    
    torch.manual_seed(42)

    return torch.randn(len(tokens), embedding_dim)