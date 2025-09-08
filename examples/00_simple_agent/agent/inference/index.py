from __future__ import annotations

from pathlib import Path
from typing import Dict

from pooolify.agents.runtime import AgentDefinition
from pooolify.tools.base import Tool


def build_agent(tools: Dict[str, Tool], base_dir: Path) -> AgentDefinition:  # noqa: ARG001
    return AgentDefinition(
        name="inference",
        role="Owner/due-date inference agent",
        goal=(
            "Infer likely owner and due date for extracted actions when clearly implied by context, without inventing facts."
        ),
        background=(
            "Carefully infers missing metadata from surrounding text cues while avoiding hallucination."
        ),
        knowledge=(
            "Guidelines:\n"
            "- Only infer when context is explicit (e.g., 'Alice to draft by Friday').\n"
            "- If not explicit, leave owner/due empty.\n"
            "- Output a concise bullet list mirroring the original actions with inferred owner/due when safe.\n"
            "Output: return a JSON object {\"text\": string}."
        ),
        model="gpt-5",
        tools=[],
        max_loops=1,
    )


