"""
Coordinator Agent -- Routes queries to the appropriate specialist.

This is the entry point for all user queries. It uses LLM reasoning
(not keyword matching) to classify queries and delegate to:
- Retriever Agent: ML/textbook knowledge questions
- General Agent: calculations, web search, general reasoning

Architecture:
    CoordinatorTeam (class)
    ├── RetrieverAgent.agent  (RAG specialist)
    └── GeneralAgent.agent   (Reasoning + Tools)

This implements genuine multi-agent orchestration via Agno's Team
with 'coordinate' mode, not simple prompt chaining.
"""
from typing import Optional

from agno.models.openrouter import OpenRouter
# from agno.models.google import Gemini #
from agno.team import Team
from agno.team.mode import TeamMode
from agno.db.sqlite import SqliteDb

from src.config import LLM_MODEL_ID, OPENROUTER_API_KEY, DATA_DIR
# from src.config import GEMINI_MODEL_ID, GEMINI_API_KEY, DATA_DIR #
from src.agents.retriever import RetrieverAgent
from src.agents.general import GeneralAgent


class CoordinatorTeam:
    """Orchestrates multiple specialist agents via Agno's Team API.

    Responsibilities:
        - Owns instances of RetrieverAgent and GeneralAgent
        - Configures the Agno Team with routing instructions
        - Provides session persistence via SQLite
        - Exposes the Team object for use by AgentOS

    Usage:
        coordinator = CoordinatorTeam()
        coordinator.team.print_response("What is cross-validation?")
    """

    def __init__(
        self,
        retriever: Optional[RetrieverAgent] = None,
        general: Optional[GeneralAgent] = None,
    ):
        self._retriever = retriever or RetrieverAgent()
        self._general = general or GeneralAgent()
        self._db = self._build_db()
        self.team = self._build_team()

    @staticmethod
    def _build_db() -> SqliteDb:
        """Create SQLite storage for session persistence."""
        db_path = str(DATA_DIR / "sessions.db")
        return SqliteDb(db_file=db_path)

    def _build_team(self) -> Team:
        """Construct the Agno Team with coordinator routing logic."""
        return Team(
            name="Research Team",
            mode=TeamMode.coordinate,
            model=OpenRouter(id=LLM_MODEL_ID, api_key=OPENROUTER_API_KEY),
            # model=Gemini(id=GEMINI_MODEL_ID, api_key=GEMINI_API_KEY),   #
            members=[
                self._retriever.agent,
                self._general.agent,
            ],
            db=self._db,
            instructions=[
                "You are the Research Assistant coordinator.",
                "Your job is to understand the user's query and delegate it to the right specialist.",
                "",
                "CLASSIFICATION (state this before delegating):",
                "Format: 'Classification: [RAG|GENERAL] -- Reason: [brief explanation]'",
                "Rate your confidence: HIGH (clear match), MEDIUM (could go either way), LOW (unsure).",
                "For LOW confidence, explain your reasoning in detail.",
                "",
                "ROUTING RULES:",
                "1. Retriever Agent (RAG) -- Use for questions about:",
                "   - Machine learning concepts (algorithms, models, evaluation)",
                "   - Topics from the ML textbook (scikit-learn, supervised/unsupervised learning)",
                "   - Technical ML explanations (cross-validation, feature engineering, etc.)",
                "",
                "2. General Agent (GENERAL) -- Use for:",
                "   - Mathematical calculations (arithmetic, conversions)",
                "   - Web searches (current events, trends, facts)",
                "   - General knowledge questions (not ML-specific)",
                "   - Coding questions (not about the textbook)",
                "",
                "IMPORTANT RULES:",
                "- Always state your classification and confidence BEFORE delegating.",
                "- If a query has both ML and general components, delegate to the most relevant agent.",
                "- Do NOT answer queries yourself -- always delegate to a specialist.",
            ],
            markdown=True,
            share_member_interactions=True,
        )
