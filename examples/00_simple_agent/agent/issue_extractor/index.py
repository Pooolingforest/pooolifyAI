from __future__ import annotations

from pathlib import Path
from typing import Dict

from pooolify.agents.runtime import AgentDefinition
from pooolify.tools.base import Tool


def build_agent(tools: Dict[str, Tool], base_dir: Path) -> AgentDefinition:  # noqa: ARG001
    return AgentDefinition(
        name="issue_extractor",
        role="Open issues extractor",
        goal=(
            "List uncertainties, blockers, risks, or unresolved questions mentioned in the notes."
        ),
        background=(
            "Analyst who identifies potential risks and unclear points from discussions."
        ),
        knowledge=(
            "Guidelines:\n"
            "- Extract only issues (not actions or decisions).\n"
            "- Keep each issue concise; one line each.\n"
            "Output: return a JSON object {\"text\": string} where text is a bullet list of issues."
        ),
        model="gpt-5",
        tools=[],
        max_loops=1,
    )


