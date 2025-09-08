from __future__ import annotations

from pathlib import Path
from typing import Dict

from pooolify.agents.runtime import AgentDefinition
from pooolify.tools.base import Tool


def build_agent(tools: Dict[str, Tool], base_dir: Path) -> AgentDefinition:  # noqa: ARG001
    return AgentDefinition(
        name="prioritizer",
        role="Priority and tag assigner",
        goal=(
            "Assign priorities (P0/P1/P2) and tags (area/owner) to actions and issues."
        ),
        background=(
            "Organizes tasks to improve execution clarity and urgency alignment."
        ),
        knowledge=(
            "Guidelines:\n"
            "- P0: urgent+critical, P1: important soon, P2: normal.\n"
            "- Tags: choose simple categories like 'backend', 'frontend', 'ops', names for owners if explicit.\n"
            "Output: return a JSON object {\"text\": string} with compact list including priority and tags."
        ),
        model="gpt-5",
        tools=[],
        max_loops=1,
    )


