from __future__ import annotations

from pathlib import Path
from typing import Dict

from pooolify.agents.runtime import AgentDefinition
from pooolify.tools.base import Tool


def build_agent(tools: Dict[str, Tool], base_dir: Path) -> AgentDefinition:  # noqa: ARG001
    return AgentDefinition(
        name="decision_extractor",
        role="Key decisions extractor",
        goal=(
            "Extract final decisions and approvals captured during the meeting with brief rationale if stated."
        ),
        background=(
            "Helps teams remember what was committed or decided with clarity."
        ),
        knowledge=(
            "Guidelines:\n"
            "- Include only explicit decisions (avoid assumptions).\n"
            "- Optional: add rationale in parentheses if present in notes.\n"
            "Output: return a JSON object {\"text\": string} where text is a bullet list of decisions."
        ),
        model="gpt-5",
        tools=[],
        max_loops=1,
    )


