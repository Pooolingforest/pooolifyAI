from pathlib import Path
from dotenv import load_dotenv

# Load example-local .env BEFORE importing pooolify to ensure DATABASE_URL is seen
load_dotenv((Path(__file__).parent / ".env"), override=False)

from fastapi import FastAPI

from pooolify.api.server import create_fastapi_app
from pooolify.core.app import PooolifyApp, ManagerLimits
from pooolify.core.autoload import load_agents, load_tools, load_manager
from pooolify.llm.base import set_provider_override


# Allow env to select the provider; do not override in example
set_provider_override(None)

BASE = Path(__file__).parent

# 1) tools -> build/load -> dict[str, Tool]
tools = load_tools(BASE, dir_name="tools")

# 2) agents -> build/load with tools -> dict[str, AgentDefinition]
agents = load_agents(BASE, dir_name="agent", tools=tools)

# 3) manager -> optional override from agent/index.py
manager_cfg = load_manager(BASE, dir_name="agent")

# 4) register all agents to the app
core = PooolifyApp(
    limits=ManagerLimits(max_steps=10, max_depth=2, max_duration_s=120),
    manager=manager_cfg,
    cors=PooolifyApp.CorsOptions(
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
        max_age=600,
    ),
)
for agent in agents.values():
    core.register_agent(agent)

app: FastAPI = create_fastapi_app(core)


