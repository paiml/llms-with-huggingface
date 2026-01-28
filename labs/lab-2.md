# Lab 2: Retrieval Augmented Generation (RAG)

In this lab, you will learn how to extend LLM capabilities using Retrieval Augmented Generation (RAG). You'll work with embeddings, vector databases, and combine search results with LLM responses to create more accurate and contextual answers.

## Learning Objectives

By the end of this lab, you will be able to:

- Understand embeddings and their role in semantic search
- Generate embeddings using Sentence Transformers
- Build and query a vector database with Qdrant
- Implement a complete RAG pipeline

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install sentence-transformers qdrant-client pandas openai
```

## Key Concepts

- **Embeddings**: Vector representations of text that capture semantic meaning
- **Sentence Transformers**: Library for generating high-quality text embeddings
- **Vector Database**: Specialized database for storing and searching embeddings
- **Semantic Search**: Finding similar content based on meaning, not just keywords
- **RAG**: Combining retrieved context with LLM generation for better answers

## Lab Exercises

### Exercise 1: Understanding Embeddings

Navigate to the [examples/2-rag](../examples/2-rag/) directory.

1. Study the concept of embeddings:

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for text
sentences = [
    "The weather is sunny today",
    "It's a beautiful day outside",
    "Python is a programming language"
]

embeddings = encoder.encode(sentences)
print(f"Embedding shape: {embeddings[0].shape}")  # 384 dimensions
```

2. Similar sentences have similar embeddings (closer in vector space)

### Exercise 2: Loading and Preparing Data

Study [rag.ipynb](../examples/2-rag/rag.ipynb):

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('top_rated_wines.csv')
df = df[df['variety'].notna()]  # Remove NaN values

# Sample data for faster processing
data = df.sample(700).to_dict('records')
print(f"Loaded {len(data)} records")
```

### Exercise 3: Setting Up the Vector Database

Create an in-memory Qdrant vector database:

```python
from qdrant_client import models, QdrantClient

# Create in-memory Qdrant instance
qdrant = QdrantClient(":memory:")

# Create a collection to store embeddings
qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # 384 for MiniLM
        distance=models.Distance.COSINE  # Use cosine similarity
    )
)
```

### Exercise 4: Indexing Data

Upload data with embeddings to the vector database:

```python
# Vectorize and upload all records
qdrant.upload_points(
    collection_name="top_wines",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["notes"]).tolist(),  # Embed the notes field
            payload=doc,  # Store the full document
        ) for idx, doc in enumerate(data)
    ]
)

print("Data indexed successfully!")
```

### Exercise 5: Semantic Search

Search for similar documents:

```python
user_prompt = "Suggest me an amazing Malbec wine from Argentina"

# Search the vector database
hits = qdrant.search(
    collection_name="top_wines",
    query_vector=encoder.encode(user_prompt).tolist(),
    limit=3  # Return top 3 results
)

# Display results
for hit in hits:
    print(f"Score: {hit.score:.4f}")
    print(f"Wine: {hit.payload}")
    print("---")
```

### Exercise 6: Combining Search with LLM

Complete the RAG pipeline:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Get search results
search_results = [hit.payload for hit in hits]

# Send to LLM with context
completion = client.chat.completions.create(
    model="qwen2.5-coder:7b-instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a wine specialist. Use the provided wine information to give recommendations."
        },
        {
            "role": "user",
            "content": user_prompt
        },
        {
            "role": "assistant",
            "content": f"Based on my search, here are some options: {search_results}"
        }
    ]
)

print(completion.choices[0].message.content)
```

### Exercise 7: Building a Complete RAG Function

Create a reusable RAG function:

```python
def rag_query(user_question, collection_name="top_wines", top_k=3):
    """
    Perform RAG: search for relevant context and generate an answer.
    """
    # Step 1: Embed the question
    query_vector = encoder.encode(user_question).tolist()

    # Step 2: Search for relevant documents
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    # Step 3: Extract context from search results
    context = "\n".join([str(hit.payload) for hit in hits])

    # Step 4: Generate answer with LLM
    completion = client.chat.completions.create(
        model="qwen2.5-coder:7b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer based on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_question}"
            }
        ]
    )

    return completion.choices[0].message.content

# Use the RAG function
answer = rag_query("What's a good wine for a special dinner?")
print(answer)
```

## Challenge

1. Create a RAG system for a different dataset (e.g., product reviews, documentation)
2. Experiment with different embedding models from Sentence Transformers
3. Implement a chat interface that uses RAG for every response
4. Add filtering to the vector search (e.g., only wines from a specific region)
5. Compare answers with and without RAG context

## Summary

In this lab, you learned how to:
- Generate embeddings using Sentence Transformers
- Set up and populate a Qdrant vector database
- Perform semantic search to find relevant documents
- Combine search results with LLM generation in a RAG pipeline
- Build reusable RAG functions for question answering

## Next Steps

Continue to [Lab 3: Building LLM APIs with FastAPI](./lab-3.md) to learn how to expose your LLM applications as web services.
