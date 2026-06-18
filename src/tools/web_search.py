"""
Web Search Tool for the AI Research Assistant.

Primary: Tavily Search API (optimized for AI agents).
Fallback: Mock search results when Tavily is unavailable.

Design Decision: Graceful degradation pattern -- external APIs can fail
(rate limits, network issues, key expiry). The mock fallback ensures the
system remains functional and demonstrable regardless of API availability.
"""
import json
import os
from typing import List, Dict, Any, Optional

from agno.tools import Toolkit


class WebSearchTools(Toolkit):
    """Web search toolkit with Tavily integration and mock fallback.

    Uses Tavily Search for real web results. If the Tavily API key is missing
    or the API fails, falls back to curated mock results.

    Responsibilities:
        - Real web search via Tavily API
        - Mock fallback for offline / demo scenarios
        - Structured JSON output for LLM consumption

    Usage:
        search = WebSearchTools()
        search.search_web("latest trends in AI agents")
    """

    # Curated mock responses for common AI/ML queries
    _MOCK_DB: Dict[str, Dict[str, Any]] = {
        "machine learning": {
            "answer": "Machine learning is a subset of AI that enables systems to learn from data.",
            "results": [
                {
                    "title": "Machine Learning - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Machine_learning",
                    "content": (
                        "Machine learning (ML) is a field of study in artificial intelligence "
                        "concerned with the development of algorithms that learn from data."
                    ),
                },
            ],
        },
        "deep learning": {
            "answer": "Deep learning is a subset of machine learning using neural networks with many layers.",
            "results": [
                {
                    "title": "Deep Learning - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Deep_learning",
                    "content": (
                        "Deep learning is part of a broader family of machine learning methods "
                        "based on artificial neural networks with representation learning."
                    ),
                },
            ],
        },
        "python": {
            "answer": "Python is a high-level, general-purpose programming language widely used in data science and AI.",
            "results": [
                {
                    "title": "Python (programming language) - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                    "content": (
                        "Python is a high-level, general-purpose programming language. "
                        "Its design philosophy emphasizes code readability."
                    ),
                },
            ],
        },
        "transformer": {
            "answer": "Transformers are a deep learning architecture based on the self-attention mechanism.",
            "results": [
                {
                    "title": "Transformer (deep learning) - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
                    "content": (
                        "A transformer is a deep learning architecture based on the multi-head "
                        "attention mechanism, proposed in the 2017 paper 'Attention Is All You Need'."
                    ),
                },
            ],
        },
    }

    def __init__(self):
        super().__init__(name="web_search")
        self._client: Optional[Any] = None
        self._tavily_available = self._check_tavily()
        self.register(self.search_web)

    def _check_tavily(self) -> bool:
        """Check if the Tavily API client can be initialized."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("[WARNING] Tavily API key not found -- using mock search fallback")
            return False
        try:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=api_key)
            return True
        except ImportError:
            print("[WARNING] tavily-python not installed -- using mock search fallback")
            return False

    def search_web(self, query: str, max_results: int = 3) -> str:
        """Search the web for information about a topic.

        Uses Tavily API when available, otherwise returns curated mock results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (default: 3).

        Returns:
            JSON string with search results containing title, URL, and content.
        """
        if self._tavily_available:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> str:
        """Execute a real search via the Tavily API."""
        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True,
            )

            results: List[Dict[str, str]] = []
            for r in response.get("results", [])[:max_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500],  # Truncate for context window
                })

            return json.dumps({
                "query": query,
                "answer": response.get("answer", ""),
                "results": results,
                "source": "tavily",
                "status": "success",
            })
        except Exception as e:
            print(f"[WARNING] Tavily search failed: {e} -- falling back to mock")
            return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> str:
        """Return curated mock search results for demo purposes."""
        query_lower = query.lower()

        # Try to find the best matching mock entry
        best_match = None
        for key, value in self._MOCK_DB.items():
            if key in query_lower:
                best_match = value
                break

        if best_match:
            return json.dumps({
                "query": query,
                "answer": best_match["answer"],
                "results": best_match["results"][:max_results],
                "source": "mock (Tavily unavailable)",
                "status": "success",
            })

        # Generic fallback for unmatched queries
        return json.dumps({
            "query": query,
            "answer": (
                f"[Mock] This is a simulated search result for: '{query}'. "
                "In production, this would return real web results via Tavily."
            ),
            "results": [
                {
                    "title": f"Search results for: {query}",
                    "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
                    "content": (
                        f"[Mock result] Information about '{query}' would appear here "
                        "with real Tavily search integration."
                    ),
                }
            ],
            "source": "mock (Tavily unavailable)",
            "status": "success",
        })
