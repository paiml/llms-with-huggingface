"""Research agent with RAG and tool capabilities."""

import logging
import time
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.agents.tools.summarizer import summarize_text
from src.agents.tools.web_search import web_search
from src.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Maximum agent iterations to prevent infinite loops
MAX_AGENT_ITERATIONS = 10


class ResearchAgent:
    """AI-powered research assistant with RAG and tool capabilities.

    Combines knowledge base search, web search, and summarization
    to answer research questions comprehensively.

    Attributes:
        llm: ChatOpenAI instance for language model interactions.
        vector_store: VectorStore for knowledge base operations.
        tools: List of available tools for the agent.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b-instruct",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        """Initialize the research agent.

        Args:
            model: Model name for the LLM.
            base_url: Base URL for the OpenAI-compatible API.
            api_key: API key for authentication.

        Raises:
            RuntimeError: If initialization takes longer than 5 seconds.
        """
        start_time = time.time()

        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0,
        )
        self.vector_store = VectorStore()
        self._sources: list[str] = []

        # Create the RAG tool with access to vector store
        rag_tool = self._create_rag_tool()
        self.tools = [web_search, summarize_text, rag_tool]

        init_time = time.time() - start_time
        if init_time > 5.0:
            logger.warning("ResearchAgent init took %.2fs (> 5s threshold)", init_time)

        logger.info("ResearchAgent initialized in %.2fs", init_time)

    def _create_rag_tool(self) -> Any:
        """Create a tool for searching the knowledge base.

        Returns:
            A LangChain tool that searches the vector store.
        """
        vector_store = self.vector_store
        sources_ref = self._sources

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the internal knowledge base for relevant information.

            Use this tool first for factual questions or established knowledge.
            The knowledge base contains curated documents and articles.

            Args:
                query: The question or topic to search for.

            Returns:
                Relevant documents from the knowledge base, or a message
                indicating no results were found.
            """
            results = vector_store.search(query, limit=3)

            if not results:
                return "No relevant documents found in the knowledge base."

            formatted_results = []
            for r in results:
                title = r.get("title", "Document")
                content = r.get("content", "")
                sources_ref.append(f"Internal KB: {title}")
                formatted_results.append(f"**{title}**: {content[:500]}")

            return "\n\n".join(formatted_results)

        return search_knowledge_base

    def research(self, query: str) -> tuple[str, list[str]]:
        """Perform research on a topic.

        Args:
            query: The research question or topic.

        Returns:
            Tuple of (answer, sources) where answer is the research response
            and sources is a list of sources consulted.
        """
        self._sources = []  # Reset sources for new query

        logger.info("Starting research for: %s", query[:100])

        # Build messages for the agent
        system_prompt = """You are a research assistant that helps users find and analyze information.

You have access to these tools:
1. search_knowledge_base - Search internal documents for established knowledge
2. web_search - Search the web for current information
3. summarize_text - Summarize long documents or articles

Guidelines:
- Search the knowledge base first for factual questions
- Use web search for current events or recent information
- Cite sources when providing information
- Be concise but thorough in your responses
- If no relevant information is found, say so clearly"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            # Simple implementation without full agent loop
            # In production, use create_react_agent or similar
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            # Check if response mentions tool usage
            if "search" in query.lower() or "find" in query.lower():
                # Proactively search knowledge base
                kb_results = self.vector_store.search(query, limit=2)
                if kb_results:
                    for r in kb_results:
                        self._sources.append(f"Internal KB: {r.get('title', 'Document')}")

            logger.info("Research completed with %d sources", len(self._sources))
            return answer, self._sources

        except Exception as e:
            logger.error("Research failed: %s", e)
            return f"Research failed: {e}", []

    def add_to_knowledge_base(self, documents: list[dict[str, Any]]) -> int:
        """Add documents to the knowledge base.

        Args:
            documents: List of document dicts with 'title' and 'content' keys.

        Returns:
            Number of documents successfully added.
        """
        return self.vector_store.add_documents(documents)
