"""
General Agent -- Handles reasoning, calculations, and web search.

This agent:
1. Answers general knowledge questions using LLM reasoning
2. Uses the Calculator tool for mathematical computations
3. Uses Web Search for current information or topics outside its training
4. Decides WHEN to use tools vs. answer directly

"""
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
# from agno.models.google import Gemini

from src.config import LLM_MODEL_ID, OPENROUTER_API_KEY
# from src.config import GEMINI_MODEL_ID, GEMINI_API_KEY
from src.tools.calculator import CalculatorTools
from src.tools.web_search import WebSearchTools


class GeneralAgent:
    """Wraps an Agno Agent with calculator and web search capabilities.

    Responsibilities:
        - Configures the agent with reasoning-focused instructions
        - Provides calculator and web search toolkits
        - Lets the LLM decide WHEN to use each tool

    Usage:
        general = GeneralAgent()
        general.agent.print_response("What is 45 * 32 + 17?")
    """

    def __init__(
        self,
        calculator: CalculatorTools = None,
        web_search: WebSearchTools = None,
    ):
        self._calculator = calculator or CalculatorTools()
        self._web_search = web_search or WebSearchTools()
        self.agent = self._build_agent()

    def _build_agent(self) -> Agent:
        """Construct the Agno Agent with tool kits and instructions."""
        return Agent(
            name="General Agent",
            role="General Reasoning and Tool Use Specialist",
            model=OpenRouter(id=LLM_MODEL_ID, api_key=OPENROUTER_API_KEY),
            # model=Gemini(id=GEMINI_MODEL_ID, api_key=GEMINI_API_KEY),
            tools=[self._calculator, self._web_search],
            instructions=[
                "You are a general-purpose reasoning assistant with access to tools.",
                "You can answer general knowledge questions directly using your training.",
                "",
                "TOOL USAGE GUIDELINES:",
                "- Math calculations -> use the calculator tool. Do NOT compute in your head.",
                "- Unit conversions -> use the calculator unit_convert tool.",
                "- Current events or uncertain facts -> use the web search tool.",
                "- Simple facts you already know -> answer directly, no tool needed.",
                "",
                "OUTPUT FORMAT:",
                "1. If you used a tool, show which tool and what input you provided.",
                "2. Present the tool result clearly.",
                "3. Then provide your final answer.",
                "4. If no tool was needed, just answer directly.",
            ],
            markdown=True,
        )
