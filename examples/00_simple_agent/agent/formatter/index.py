from __future__ import annotations

from pathlib import Path
from typing import Dict

from pooolify.agents.runtime import AgentDefinition
from pooolify.tools.base import Tool


def build_agent(tools: Dict[str, Tool], base_dir: Path) -> AgentDefinition:  # noqa: ARG001
    return AgentDefinition(
        name="formatter",
        role="Final formatter",
        goal=(
            "Combine extractions, inference, and prioritization into a clean, concise summary."
        ),
        background=(
            "Turns heterogeneous intermediate outputs into a crisp, user-friendly final text."
        ),
        knowledge=(
            "Guidelines:\n"
            "- Sections: Actions, Issues, Decisions.\n"
            "- Use compact bullet points; keep under ~200 words.\n"
            "- If some sections are empty, omit them.\n"
            "Output: return a JSON object {\"text\": string} containing the final summary."
        ),
        model="gpt-5",
        tools=[],
        max_loops=1,
    )


