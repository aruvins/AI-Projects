from src.embed import generate_embeddings
from src.build_index import build_index
from src.search import search_loop


def main():

    print("=" * 50)
    print("VECTOR DATABASE SEARCH ENGINE")
    print("=" * 50)

    print("\n[1/3] Generating Embeddings...")
    generate_embeddings()

    print("\n[2/3] Building FAISS Index...")
    build_index()

    print("\n[3/3] Starting Search Engine...")
    search_loop()


if __name__ == "__main__":
    main()