# Capstone Project: AI-Powered Research Assistant

Build a comprehensive AI-powered research assistant that combines all the concepts learned throughout this course. This project will demonstrate your mastery of LLM integration, RAG systems, API development, and agentic workflows.

## Project Overview

You will build a **Research Assistant** that can:
- Answer questions using a knowledge base (RAG)
- Search the web for current information (Agents)
- Summarize documents and articles (LLM APIs)
- Provide structured responses for data analysis (Structured Output)
- Run as a web service accessible via API (FastAPI)

## Learning Objectives

By completing this capstone project, you will demonstrate your ability to:

- Integrate multiple LLM capabilities into a cohesive application
- Design and implement a production-ready RAG pipeline
- Build agentic workflows with custom tools
- Create robust APIs for LLM-powered features
- Handle errors gracefully and provide meaningful feedback
- Structure code for maintainability and extensibility

## Prerequisites

Complete all five labs before starting this project:

- [Lab 1: Building a Simple Chat Interface](../labs/lab-1.md)
- [Lab 2: Retrieval Augmented Generation (RAG)](../labs/lab-2.md)
- [Lab 3: Building LLM APIs with FastAPI](../labs/lab-3.md)
- [Lab 4: Working with Small Language Models](../labs/lab-4.md)
- [Lab 5: Building Agentic Applications](../labs/lab-5.md)

### Required Dependencies

```bash
pip install openai fastapi uvicorn langchain langchain-openai \
    sentence-transformers qdrant-client pandas pydantic httpx
```

## Project Architecture

```
research-assistant/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py          # Chat endpoints
│   │       ├── search.py        # RAG search endpoints
│   │       └── research.py      # Agent endpoints
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── research_agent.py    # Main research agent
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── web_search.py    # Web search tool
│   │       ├── summarizer.py    # Document summarization
│   │       └── calculator.py    # Math calculations
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Embedding generation
│   │   ├── vectorstore.py       # Vector database operations
│   │   └── retriever.py         # Document retrieval
│   └── models/
│       ├── __init__.py
│       ├── schemas.py           # Pydantic models
│       └── llm.py               # LLM client configuration
├── data/
│   └── knowledge_base/          # Documents for RAG
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Implementation Guide

### Part 1: Core LLM Integration (From Lab 1)

Create a reusable LLM client that supports both local and remote models:

```python
# src/models/llm.py
import os
from typing import Any
from openai import OpenAI
from openai.types.chat import ChatCompletion

class LLMClient:
    """Unified LLM client for local and remote models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=base_url or os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        )
        self.model = model or os.getenv("MODEL_NAME", "qwen2.5-coder:7b-instruct")

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> ChatCompletion:
        """Send a chat completion request."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_response(self, user_message: str, system_prompt: str = "") -> str:
        """Get a simple text response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        completion = self.chat(messages)
        return completion.choices[0].message.content or ""
```

### Part 2: RAG Pipeline (From Lab 2)

Implement the knowledge base with vector search:

```python
# src/rag/vectorstore.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from typing import Any

class VectorStore:
    """Vector database for document storage and retrieval."""

    def __init__(self, collection_name: str = "knowledge_base"):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")  # Use persistent storage in production
        self.collection_name = collection_name
        self._create_collection()

    def _create_collection(self) -> None:
        """Initialize the vector collection."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        points = [
            models.PointStruct(
                id=idx,
                vector=self.encoder.encode(doc["content"]).tolist(),
                payload=doc,
            )
            for idx, doc in enumerate(documents)
        ]
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for relevant documents."""
        query_vector = self.encoder.encode(query).tolist()
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        return [hit.payload for hit in hits.points]
```

### Part 3: Agent Tools (From Lab 5)

Create tools for the research agent:

```python
# src/agents/tools/web_search.py
import httpx
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic.

    Args:
        query: The search query to look up.

    Returns:
        Search results as a formatted string.
    """
    # In production, use a real search API (Google, Bing, etc.)
    # This is a mock implementation for demonstration
    return f"Mock search results for: {query}"


# src/agents/tools/summarizer.py
from langchain_core.tools import tool

@tool
def summarize_text(text: str, max_length: int = 200) -> str:
    """Summarize a long piece of text into key points.

    Args:
        text: The text to summarize.
        max_length: Maximum length of the summary.

    Returns:
        A concise summary of the text.
    """
    # Use the LLM to generate a summary
    from src.models.llm import LLMClient

    client = LLMClient()
    prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
    return client.get_response(prompt, system_prompt="You are a concise summarizer.")
```

### Part 4: Research Agent (From Lab 5)

Build the main agent that orchestrates all capabilities:

```python
# src/agents/research_agent.py
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.agents.tools.web_search import web_search
from src.agents.tools.summarizer import summarize_text
from src.rag.vectorstore import VectorStore

class ResearchAgent:
    """AI-powered research assistant with RAG and tool capabilities."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen2.5-coder:7b-instruct",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0,
        )
        self.vector_store = VectorStore()
        self.tools = [web_search, summarize_text, self._create_rag_tool()]
        self.agent = self._create_agent()

    def _create_rag_tool(self):
        """Create a tool for searching the knowledge base."""
        from langchain_core.tools import tool

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the internal knowledge base for relevant information.

            Args:
                query: The question or topic to search for.

            Returns:
                Relevant documents from the knowledge base.
            """
            results = self.vector_store.search(query, limit=3)
            if not results:
                return "No relevant documents found in the knowledge base."
            return "\n\n".join([
                f"**{r.get('title', 'Document')}**: {r.get('content', '')}"
                for r in results
            ])

        return search_knowledge_base

    def _create_agent(self):
        """Create the LangChain agent."""
        system_prompt = """You are a research assistant that helps users find and analyze information.

You have access to:
1. search_knowledge_base - Search internal documents for established knowledge
2. web_search - Search the web for current information
3. summarize_text - Summarize long documents or articles

Always:
- Search the knowledge base first for factual questions
- Use web search for current events or recent information
- Provide sources and citations when possible
- Be concise but thorough in your responses"""

        return create_agent(self.llm, self.tools, system_prompt=system_prompt)

    def research(self, query: str) -> str:
        """Perform research on a topic."""
        result = self.agent.invoke({"messages": [("user", query)]})
        return result["messages"][-1].content

    def add_to_knowledge_base(self, documents: list[dict]) -> None:
        """Add documents to the knowledge base."""
        self.vector_store.add_documents(documents)
```

### Part 5: FastAPI Application (From Lab 3)

Create the web API:

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.research_agent import ResearchAgent
from src.models.llm import LLMClient

app = FastAPI(
    title="Research Assistant API",
    description="AI-powered research assistant with RAG and agent capabilities",
    version="1.0.0",
)

# Initialize components
research_agent = ResearchAgent()
llm_client = LLMClient()


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str
    system_prompt: str = "You are a helpful assistant."


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    response: str


class ResearchRequest(BaseModel):
    """Request body for research endpoint."""
    query: str


class ResearchResponse(BaseModel):
    """Response body for research endpoint."""
    answer: str
    sources: list[str] = []


class DocumentRequest(BaseModel):
    """Request body for adding documents."""
    title: str
    content: str


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Simple chat endpoint."""
    try:
        response = llm_client.get_response(
            request.message,
            system_prompt=request.system_prompt,
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    """Research endpoint using the agent."""
    try:
        answer = research_agent.research(request.query)
        return ResearchResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
def add_document(request: DocumentRequest):
    """Add a document to the knowledge base."""
    try:
        research_agent.add_to_knowledge_base([{
            "title": request.title,
            "content": request.content,
        }])
        return {"status": "success", "message": "Document added to knowledge base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Evaluation Criteria

Your capstone project will be evaluated on:

### Functionality (40 points)
- [ ] Chat endpoint works correctly (10 pts)
- [ ] RAG search returns relevant results (10 pts)
- [ ] Agent uses tools appropriately (10 pts)
- [ ] Documents can be added to knowledge base (10 pts)

### Code Quality (30 points)
- [ ] Clean, well-organized code structure (10 pts)
- [ ] Type hints and docstrings (10 pts)
- [ ] Error handling and edge cases (10 pts)

### Documentation (20 points)
- [ ] Clear README with setup instructions (10 pts)
- [ ] API documentation (Swagger/OpenAPI) (10 pts)

### Bonus Features (10 points)
- [ ] Streaming responses (5 pts)
- [ ] Conversation memory (5 pts)

## Challenge Extensions

Once you've completed the basic project, try these extensions:

1. **Add Authentication**: Implement API key authentication for your endpoints

2. **Persistent Storage**: Replace in-memory vector store with persistent storage (Qdrant, Pinecone, or Chroma)

3. **Streaming Responses**: Implement Server-Sent Events (SSE) for streaming LLM responses

4. **Conversation Memory**: Add Redis or database-backed conversation history

5. **Multi-Model Support**: Allow switching between different LLM providers

6. **Rate Limiting**: Add request throttling to prevent abuse

7. **Monitoring**: Add logging and metrics with OpenTelemetry

8. **Deployment**: Deploy to a cloud provider with Docker and Kubernetes

## Submission Guidelines

1. Create a new repository for your project
2. Include a comprehensive README with:
   - Project description
   - Setup instructions
   - API documentation
   - Example usage
3. Ensure all tests pass
4. Document any additional features you implemented

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## Summary

This capstone project brings together all the skills you've learned:

| Lab | Skill | Application in Capstone |
|-----|-------|------------------------|
| Lab 1 | Chat Interface | Core LLM client and chat endpoint |
| Lab 2 | RAG | Knowledge base and document retrieval |
| Lab 3 | FastAPI | Web API for all features |
| Lab 4 | Small Models | Efficient local model usage |
| Lab 5 | Agents | Research agent with tools |

Congratulations on completing the course! You now have the skills to build production-ready LLM applications.
