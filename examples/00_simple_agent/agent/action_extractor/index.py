from __future__ import annotations

from pathlib import Path
from typing import Dict

from pooolify.agents.runtime import AgentDefinition
from pooolify.tools.base import Tool


def build_agent(tools: Dict[str, Tool], base_dir: Path) -> AgentDefinition:  # noqa: ARG001
    return AgentDefinition(
        name="action_extractor",
        role="Action items extractor",
        goal=(
            "Extract concrete, actionable tasks from meeting notes. Include verb, owner if given, and any explicit due dates."
        ),
        background=(
            "Expert at skimming long meeting notes and listing clear action items only."
        ),
        knowledge=(
            "Guidelines:\n"
            "- Focus on actions to be taken (verbs like implement, fix, decide, draft).\n"
            "- Prefer short titles; keep each action to one line.\n"
            "- If owner/date are not in the notes, omit them (do not invent).\n"
            "Output: return a JSON object {\"text\": string} where text is a bullet list of actions."
        ),
        model="gpt-5",
        tools=[],
        max_loops=1,
    )


