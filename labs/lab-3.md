# Lab 3: Building LLM APIs with FastAPI

In this lab, you will learn how to build HTTP APIs that expose LLM functionality using FastAPI. You'll create endpoints for chat completion and integrate RAG capabilities into a web service.

## Learning Objectives

By the end of this lab, you will be able to:

- Set up a FastAPI application for LLM interactions
- Create API endpoints for chat completion
- Integrate RAG with a web API
- Test and interact with your API

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install fastapi uvicorn openai sentence-transformers qdrant-client pandas
```

## Key Concepts

- **FastAPI**: Modern, fast Python web framework for building APIs
- **Pydantic**: Data validation using Python type annotations
- **Endpoint**: A URL path that handles specific requests
- **POST Request**: HTTP method for sending data to the server
- **Request Body**: Data sent with a POST request

## Lab Exercises

### Exercise 1: Basic FastAPI Setup

Navigate to the [examples/3-api](../examples/3-api/) directory.

1. Study [chat.py](../examples/3-api/chat.py):

```python
from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai

# Configure OpenAI client
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")

# Create FastAPI app
app = FastAPI()

# Define request body schema
class Body(BaseModel):
    text: str
```

### Exercise 2: Creating a Chat Endpoint

Build a simple chat endpoint:

```python
def ai_chat(user_message):
    """Send a message to the LLM and get a response."""
    message_text = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
    )
    return completion


@app.post('/generate')
def generate(body: Body):
    """API endpoint for chat generation."""
    completion = ai_chat(body.text)
    return {"text": completion['choices'][0]['message']['content']}
```

### Exercise 3: Running the API

Start the FastAPI server:

```bash
export MODEL_NAME="qwen2.5-coder:7b-instruct"
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
uvicorn chat:app --reload --port 8000
```

Test with curl:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Python?"}'
```

### Exercise 4: Integrating RAG with the API

Study [solution.py](../examples/3-api/solution.py) for a complete RAG API:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from openai import OpenAI
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Initialize components
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
)

encoder = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(":memory:")

# Load and index data on startup
df = pd.read_csv('top_rated_wines.csv')
data = df.sample(700).to_dict('records')

qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)

qdrant.upload_points(
    collection_name="top_wines",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["notes"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data)
    ]
)
```

### Exercise 5: RAG Endpoint Implementation

Add the RAG-powered endpoint:

```python
class Body(BaseModel):
    text: str


def ai_chat(user_message, extra_context=""):
    """Chat with optional RAG context."""
    user_content = user_message
    if extra_context:
        user_content = f"{user_message}\n\nHere is some relevant information:\n{extra_context}"

    message_text = [
        {
            "role": "system",
            "content": "You are a wine specialist. Help users find the best wine."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=message_text,
        temperature=0.5,
        max_tokens=400,
    )
    return completion


@app.post('/generate')
def generate(body: Body):
    """RAG-powered generation endpoint."""
    # Search for relevant context
    hits = qdrant.query_points(
        collection_name="top_wines",
        query=encoder.encode(body.text).tolist(),
        limit=4
    )

    # Extract search results
    search_results = [hit.payload for hit in hits.points]

    # Generate response with context
    completion = ai_chat(body.text, str(search_results))

    return {"text": completion.choices[-1].message.content}
```

### Exercise 6: Adding More Endpoints

Extend your API with additional functionality:

```python
@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "healthy", "model": model_name}


@app.get("/search")
def search(query: str, limit: int = 5):
    """Direct vector search endpoint."""
    hits = qdrant.query_points(
        collection_name="top_wines",
        query=encoder.encode(query).tolist(),
        limit=limit
    )
    return {"results": [hit.payload for hit in hits.points]}


class ChatHistory(BaseModel):
    messages: list[dict]


@app.post('/chat')
def chat(history: ChatHistory):
    """Multi-turn chat endpoint."""
    completion = client.chat.completions.create(
        model=model_name,
        messages=history.messages,
        temperature=0.7,
    )
    return {"response": completion.choices[0].message.content}
```

### Exercise 7: Testing Your API

Use the interactive docs or Python requests:

```python
import requests

# Test the generate endpoint
response = requests.post(
    "http://localhost:8000/generate",
    json={"text": "Recommend a red wine for steak"}
)
print(response.json())

# Test the search endpoint
response = requests.get(
    "http://localhost:8000/search",
    params={"query": "fruity white wine", "limit": 3}
)
print(response.json())
```

Or visit `http://localhost:8000/docs` for the Swagger UI.

## Challenge

1. Add authentication to your API using API keys
2. Implement rate limiting to prevent abuse
3. Add an endpoint that returns streaming responses
4. Create a simple HTML frontend that calls your API
5. Add logging to track API usage and response times

## Summary

In this lab, you learned how to:
- Set up a FastAPI application for LLM interactions
- Create POST endpoints with Pydantic validation
- Integrate RAG capabilities into a web API
- Initialize and index data on application startup
- Test APIs using curl and Python requests

## Next Steps

Continue to [Lab 4: Working with Small Language Models](./lab-4.md) to learn about efficient local models for resource-constrained environments.
