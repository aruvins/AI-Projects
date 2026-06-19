from utils.vector_store import VectorStore

class DocumentQA:
    def __init__(self, chunks):
        self.store = VectorStore()
        self.store.add_document(chunks)

    def answer(self, question):
        results = self.store.search(question, top_k=3)

        return "\n\n".join(results)