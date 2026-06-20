from sentence_transformers import SentenceTransformer
import faiss


def search_loop():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    index = faiss.read_index(
        "indexes/faiss.index"
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

    while True:

        query = input(
            "\nAsk a question ('quit' to exit): "
        )

        if query.lower() == "quit":
            break

        query_vector = model.encode(
            [query]
        )

        distances, indices = index.search(
            query_vector,
            k=1
        )

        print("\nTop Results:\n")

        for rank, idx in enumerate(
            indices[0],
            start=1
        ):

            print(
                f"{rank}. {documents[idx]}"
            )


if __name__ == "__main__":
    search_loop()