"""
Retriever Agent -- Handles RAG queries against the ML textbook.

This agent:
1. Searches the DuckDB knowledge base for relevant chunks
2. Generates answers grounded in retrieved context
3. ALWAYS shows the retrieved context (assignment requirement)
"""
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
# from agno.models.google import Gemini

from src.config import LLM_MODEL_ID, OPENROUTER_API_KEY
# from src.config import GEMINI_MODEL_ID, GEMINI_API_KEY
from src.knowledge.pdf_knowledge import KnowledgeBase, get_knowledge_base


class RetrieverAgent:
    """Wraps an Agno Agent with RAG capabilities over the ML textbook.

    Responsibilities:
        - Owns a KnowledgeBase instance for vector search
        - Exposes a tool function that the LLM can call
        - Configures the agent with retrieval-focused instructions

    Usage:
        retriever = RetrieverAgent()
        retriever.agent.print_response("What is cross-validation?")
    """

    def __init__(self, knowledge_base: KnowledgeBase = None):
        self._kb = knowledge_base or get_knowledge_base()
        self.agent = self._build_agent()

    def knowledge_search(self, query: str) -> str:
        """Search the ML textbook knowledge base and return formatted context.

        This method is registered as a tool that the LLM decides when to call.

        Args:
            query: The user's question to search for in the textbook.

        Returns:
            Formatted retrieved chunks with page numbers and relevance scores.
        """
        results = self._kb.search(query)
        return KnowledgeBase.format_results(results)

    def _build_agent(self) -> Agent:
        """Construct the Agno Agent with retrieval tool and instructions."""
        return Agent(
            name="Retriever Agent",
            role="ML Knowledge Specialist",
            model=OpenRouter(id=LLM_MODEL_ID, api_key=OPENROUTER_API_KEY),
            # model=Gemini(id=GEMINI_MODEL_ID, api_key=GEMINI_API_KEY),
            tools=[self.knowledge_search],
            instructions=[
                "You are a Machine Learning knowledge specialist.",
                "You answer questions using the 'Introduction to Machine Learning with Python' textbook.",
                "",
                "WORKFLOW:",
                "1. ALWAYS call the knowledge_search tool first with the user's query.",
                "2. Read the retrieved chunks carefully.",
                "3. Generate your answer based ONLY on the retrieved context.",
                "",
                "OUTPUT FORMAT:",
                "1. Start with a 'Retrieved Context' section showing the actual chunks you received.",
                "   Format each chunk as a blockquote with its page number.",
                "2. Then provide your answer under an 'Answer' section.",
                "3. If the knowledge base has no relevant information, say so explicitly.",
                "",
                "Be precise and cite page numbers when possible.",
            ],
            markdown=True,
        )
