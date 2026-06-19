# 🟡 Document Q&A System

A Retrieval-Augmented Generation (RAG) style NLP project that allows users to ask questions about documents using semantic search and vector embeddings.

This project demonstrates how modern AI systems retrieve relevant information from large collections of text before generating answers. Rather than relying on keyword matching, the system uses embeddings and similarity search to find document sections that are semantically related to a user's question.

---

# 🎯 Project Goals

This project is designed to teach the foundations of modern retrieval systems and document question-answering.

By completing this project, you will learn:

* Text chunking
* Semantic embeddings
* Vector representations of language
* Similarity search
* Cosine similarity
* Information retrieval
* Retrieval-Augmented Generation (RAG)
* How systems like ChatGPT Document Upload and NotebookLM work

---

# 🏗 Project Architecture

```text
document_qa/
├── data/
│   └── document.txt
│
├── utils/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── qa.py
│
├── main.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 05-natural-language-processing/project_5_document_qa
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate

```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Run the Project
```bash
python main.py
```

---

# 🔄 Project Pipeline

```text
Document
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
    ↓
User Question
    ↓
Question Embedding
    ↓
Cosine Similarity
    ↓
Top-K Retrieval
    ↓
Answer
```

---
# 🧠 Core AI Concepts

## 1. The Challenge of Question Answering

Suppose a document contains:

```text
The Eiffel Tower is located in Paris.

Paris is the capital of France.
```

A user asks:

```text
Where is the Eiffel Tower?
```

A traditional search engine might look for exact keyword matches.

Modern AI systems instead attempt to understand meaning.

The goal becomes:

```text
Find the section of the document that is
most semantically similar to the question.
```

This is known as:

```text
Semantic Search
```

---

## 2. Embeddings

The first major concept in this project is embeddings.

An embedding is a numerical representation of text.

Instead of storing text as:

```text
"The Eiffel Tower is in Paris"
```

the model converts it into:

```text
[0.23, -0.44, 0.91, ...]
```

a high-dimensional vector.

---

### Why Embeddings Work

Neural language models learn to place similar concepts near one another in a mathematical space called an embedding space.

Example:

```text
Paris
London
Berlin
```

will appear close together.

Likewise:

```text
Dog
Cat
Puppy
```

will appear near each other.

This allows the model to compare meanings rather than exact words.

---

### Embedding Space

Conceptually:

```text
Embedding Space

         Cat

    Dog

                         Car

                              Truck


Paris
     London

          Berlin
```

Semantically related ideas cluster together.

---

## 3. Document Chunking

Large documents cannot be searched efficiently as a single block of text.

Instead, documents are divided into smaller sections called chunks.

Example:

```text
100 Page Report
```

becomes:

```text
Chunk 1
Chunk 2
Chunk 3
...
Chunk N
```

---

### Why Chunking Is Necessary

Without chunking:

```text
Question
    ↓
Entire Document
```

The retrieval system must compare the question against thousands of words.

With chunking:

```text
Question
    ↓
Individual Chunks
```

The search becomes significantly faster and more accurate.

Chunking is one of the most important design decisions in modern RAG systems.

---

## 4. Creating a Vector Database

After chunking:

```text
Chunk 1
Chunk 2
Chunk 3
```

each chunk is converted into an embedding.

Result:

```text
Chunk 1 → Vector
Chunk 2 → Vector
Chunk 3 → Vector
```

These vectors form a searchable knowledge base.

This collection of vectors is often called a:

```text
Vector Store
```

or

```text
Vector Database
```

---

## 5. Similarity Search

When a user asks a question:

```text
Where is the Eiffel Tower located?
```

the question is also converted into an embedding.

```text
Question
      ↓
Embedding
```

The system then compares the question embedding against every chunk embedding.

The objective:

```text
Find the closest chunk.
```

---

## 6. Cosine Similarity

The most common similarity metric is cosine similarity.

Formula:

```text
       A · B
-------------------
||A|| × ||B||
```

where:

```text
A = Question Vector
B = Chunk Vector
```

---

### Interpretation

```text
1.0  = Identical Direction
0.0  = Unrelated
-1.0 = Opposite Direction
```

Example:

```text
Chunk 1 → 0.82
Chunk 2 → 0.14
Chunk 3 → 0.93
```

The system retrieves:

```text
Chunk 3
```

because it is most similar to the user's question.

---

## 7. Retrieval

The retrieval step identifies the most relevant document sections.

Pipeline:

```text
Question
    ↓
Embedding
    ↓
Similarity Search
    ↓
Top Matching Chunks
```

This process is known as:

```text
Information Retrieval
```

---

## 8. Top-K Retrieval

Rather than retrieving a single chunk, modern systems usually retrieve several.

Example:

```text
Top 3 Chunks
```

instead of:

```text
Top 1 Chunk
```

This improves context quality and enables multi-hop reasoning.

Example:

Chunk A:

```text
Paris is the capital of France.
```

Chunk B:

```text
France is located in Europe.
```

Question:

```text
Where is the capital of France located?
```

Neither chunk alone contains the answer.

Together:

```text
Paris, the capital of France,
is located in Europe.
```

---

## 9. Retrieval-Augmented Generation (RAG)

Modern AI systems combine retrieval with large language models.

Pipeline:

```text
Question
      ↓
Embedding
      ↓
Similarity Search
      ↓
Relevant Chunks
      ↓
LLM
      ↓
Answer
```

This architecture is called:

```text
Retrieval-Augmented Generation
```

or

```text
RAG
```

---

### Why RAG Matters

Large Language Models cannot memorize every document.

Instead:

```text
Retrieve Information
      +
Generate Response
```

Benefits:

* Better factual accuracy
* Reduced hallucinations
* Smaller context windows
* Access to private documents
* Improved scalability

Many modern AI products use this architecture.

---

# 🚀 Example

Document:

```text
The Eiffel Tower is located in Paris.

Paris is the capital of France.

France is located in Europe.
```

Question:

```text
Where is the Eiffel Tower?
```

Retrieved Context:

```text
The Eiffel Tower is located in Paris.
```

Question:

```text
Where is the capital of France located?
```

Retrieved Context:

```text
Paris is the capital of France.

France is located in Europe.
```

---

# 📈 Future Improvements

Possible extensions include:

### Better Chunking

* Sliding windows
* Recursive chunking
* Sentence-based chunking

### Better Retrieval

* FAISS
* ChromaDB
* Pinecone
* Weaviate

### Better Ranking

* Cross-encoders
* Reranking models

### Generative AI

* Local LLM integration
* Open-source LLMs
* Full RAG pipeline

### Multi-Document Support

* PDF ingestion
* Word documents
* Knowledge bases

---

# 🎓 Learning Outcomes

After completing this project, you should understand:

* How text becomes vectors
* How embeddings capture meaning
* How semantic search works
* How similarity search retrieves information
* How vector databases operate
* How modern retrieval systems function
* How RAG architectures power document assistants

This project serves as an introduction to one of the most important concepts in modern NLP and Generative AI: teaching language models how to retrieve relevant information before generating answers.

---

# 📚 References

## Embeddings

Sentence Transformers Documentation

https://www.sbert.net/

---

Reimers & Gurevych (2019)

"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

https://arxiv.org/abs/1908.10084

---

## Transformers

Vaswani et al. (2017)

"Attention Is All You Need"

https://arxiv.org/abs/1706.03762

---

## Information Retrieval

Manning, Raghavan, and Schütze

"Introduction to Information Retrieval"

https://nlp.stanford.edu/IR-book/

---

## Retrieval-Augmented Generation

Lewis et al. (2020)

"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

https://arxiv.org/abs/2005.11401

---

## Vector Search

Johnson, Douze, and Jégou (2017)

"FAISS: A Library for Efficient Similarity Search"

https://arxiv.org/abs/1702.08734

---

## Cosine Similarity

Scikit-Learn Documentation

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

---

## Modern Retrieval Systems

NotebookLM

https://notebooklm.google.com/

OpenAI Retrieval Concepts

https://platform.openai.com/docs
