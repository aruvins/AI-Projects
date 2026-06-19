from utils.chunking import chunk_text
from utils.qa import DocumentQA

with open("data/document.txt", "r",encoding="utf-8") as file:
    document = file.read()

chunks = chunk_text(document, chunk_size=128)
qa = DocumentQA(chunks)

while True:
    question = input("\nAsk a question: ")
    answer = qa.answer(question)

    print("\nRetreived Context:\n")
    print(answer)

