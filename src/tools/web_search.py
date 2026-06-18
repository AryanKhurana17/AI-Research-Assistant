"""
Web Search Tool for the AI Research Assistant.

Primary: Tavily Search API (optimized for AI agents).
Fallback: Mock search results when Tavily is unavailable.

"""
import json
import os
from typing import List, Dict, Any, Optional

from agno.tools import Toolkit

from src.logging_config import get_logger

logger = get_logger("tools.web_search")


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
        "iris": {
            "answer": "The standard Iris dataset contains 4 features: sepal length, sepal width, petal length, and petal width.",
            "results": [
                {
                    "title": "Iris flower data set - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Iris_flower_data_set",
                    "content": (
                        "The Iris dataset is a multivariate data set introduced by Ronald Fisher in 1936. "
                        "It consists of 50 samples from each of three species of Iris. Four features were "
                        "measured from each sample: the length and the width of the sepals and petals, in centimeters."
                    ),
                },
            ],
        },
        "large language model": {
            "answer": "Recent developments in LLMs include massive context windows (up to 2M tokens), multimodal native processing, advanced reasoning via reinforcement learning, and high-performance open-weight models like Llama 3.3.",
            "results": [
                {
                    "title": "Latest Trends in LLMs - TechCrunch",
                    "url": "https://techcrunch.com/latest-trends-llms",
                    "content": (
                        "The LLM landscape has shifted towards reasoning-oriented architectures, native multimodality, "
                        "and highly efficient small models that can run on edge devices."
                    ),
                },
            ],
        },
        "nvidia": {
            "answer": "NVIDIA Corporation (NVDA) is trading around $135 USD, driven by high demand for its Hopper and Blackwell AI GPUs.",
            "results": [
                {
                    "title": "NVIDIA (NVDA) Stock Price & News - Google Finance",
                    "url": "https://www.google.com/finance/quote/NVDA:NASDAQ",
                    "content": (
                        "Get the latest NVIDIA Corp (NVDA) real-time stock quote, historical performance, charts, "
                        "financial news, and analysis."
                    ),
                },
            ],
        },
        "exchange rate": {
            "answer": "The USD to INR exchange rate is hovering around 83.50 INR per US Dollar.",
            "results": [
                {
                    "title": "USD to INR Exchange Rate - XE.com",
                    "url": "https://www.xe.com/currencyconverter/convert/?Amount=1&From=USD&To=INR",
                    "content": (
                        "Convert 1 USD to INR with the XE Currency Converter. Live rates and historical charts."
                    ),
                },
            ],
        },
        "iran": {
            "answer": "Geopolitical tensions between the US and Iran continue to influence oil markets and regional security, with diplomatic efforts ongoing.",
            "results": [
                {
                    "title": "US-Iran Geopolitical Updates - Reuters",
                    "url": "https://www.reuters.com/world/middle-east/us-iran-relations",
                    "content": (
                        "Live coverage and investigative reporting on diplomatic relations, sanctions, "
                        "and security events in the Middle East."
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
            logger.warning("Tavily API key not found -- using mock search fallback")
            return False
        try:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=api_key)
            logger.info("Tavily API client initialized successfully")
            return True
        except ImportError:
            logger.warning("tavily-python not installed -- using mock search fallback")
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
        logger.info("search_web() called with query: %s", query)
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

            logger.info("Tavily search returned %d results for: %s", len(results), query)
            return json.dumps({
                "query": query,
                "answer": response.get("answer", ""),
                "results": results,
                "source": "tavily",
                "status": "success",
            })
        except Exception as e:
            logger.error("Tavily search failed for '%s': %s -- falling back to mock", query, e)
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
