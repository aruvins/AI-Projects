from sentence_transformers import SentenceTransformer
import numpy as np


def generate_embeddings():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    with open(
        "data/documents.txt",
        "r",
        encoding="utf-8"
    ) as f:

        documents = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    embeddings = model.encode(
        documents,
        convert_to_numpy=True
    )

    np.save(
        "embeddings/vectors.npy",
        embeddings
    )

    print(
        f"Generated embeddings for {len(documents)} documents"
    )


if __name__ == "__main__":
    generate_embeddings()