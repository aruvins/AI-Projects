import faiss
import numpy as np


def build_index():

    vectors = np.load(
        "embeddings/vectors.npy"
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(vectors)

    faiss.write_index(
        index,
        "indexes/faiss.index"
    )

    print(
        f"Indexed {index.ntotal} documents"
    )


if __name__ == "__main__":
    build_index()