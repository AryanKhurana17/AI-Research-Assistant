"""
AI Research Assistant -- AgentOS Application.

Exposes the multi-agent research team as an HTTP API
accessible via the Agno Agent UI or any HTTP client.

Run with:
    python -m src.app
    # OR
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from src.logging_config import setup_logging
setup_logging()

from agno.os import AgentOS

from src.agents.coordinator import CoordinatorTeam
from src.agents.retriever import RetrieverAgent
from src.agents.general import GeneralAgent


# Create agent instances (shared between team and standalone access)
retriever = RetrieverAgent()
general = GeneralAgent()
coordinator = CoordinatorTeam(retriever=retriever, general=general)

# Create AgentOS -- exposes all entities via FastAPI
agent_os = AgentOS(
    agents=[retriever.agent, general.agent],
    teams=[coordinator.team],
)

# Get the FastAPI app
app = agent_os.get_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
