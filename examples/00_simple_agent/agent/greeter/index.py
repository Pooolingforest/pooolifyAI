from __future__ import annotations

from pathlib import Path
from typing import Dict

from pooolify.agents.runtime import AgentDefinition
from pooolify.tools.base import Tool


def build_agent(tools: Dict[str, Tool], base_dir: Path) -> AgentDefinition:  # noqa: ARG001
    # Deprecated placeholder to avoid accidental loading conflicts.
    # Keeping file to preserve directory structure; manager does not schedule this agent.
    return AgentDefinition(
        name="greeter_deprecated",
        role="Deprecated placeholder",
        goal="Do not use.",
        background="",
        knowledge="",
        model="gpt-5",
        tools=[],
        max_loops=1,
    )


