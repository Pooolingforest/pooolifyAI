from __future__ import annotations

from pathlib import Path

from pooolify.core.app import PooolifyApp


def build_manager(base_dir: Path) -> PooolifyApp.ManagerConfig:  # noqa: ARG001
    return PooolifyApp.ManagerConfig(
        model="gpt-5",
        system_instruction=(
            "You are the manager agent.\n"
            "- Read the user's request, decide whether a specialist agent is needed,\n"
            "- If needed, schedule that agent to run, otherwise answer directly.\n"
            "- Keep responses short and actionable."
        ),
    )

manager = build_manager(Path(__file__).parent)


