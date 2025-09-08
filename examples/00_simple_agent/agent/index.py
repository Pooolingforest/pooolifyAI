from __future__ import annotations

from pathlib import Path

from pooolify.core.app import PooolifyApp


def build_manager(base_dir: Path) -> PooolifyApp.ManagerConfig:  # noqa: ARG001
    return PooolifyApp.ManagerConfig(
        model="gpt-5",
        system_instruction=(
            "You are the manager orchestrating a fixed multi-agent pipeline for meeting notes.\n"
            "Always set isRunAgent=true and schedule the following steps strictly:\n"
            "Step 1 (PARALLEL):\n"
            "  - action_extractor: instruction='Extract actionable tasks from the user's notes.'\n"
            "  - issue_extractor: instruction='Extract open issues/risks/questions.'\n"
            "  - decision_extractor: instruction='Extract explicit decisions.'\n"
            "Step 2 (ISOLATED):\n"
            "  - inference (dependsOn: action_extractor): instruction='Infer owner/due only when explicit.'\n"
            "Step 3 (ISOLATED):\n"
            "  - prioritizer (dependsOn: inference, issue_extractor): instruction='Assign priorities and tags.'\n"
            "Step 4 (ISOLATED):\n"
            "  - formatter (dependsOn: action_extractor, inference, issue_extractor, decision_extractor, prioritizer): instruction='Produce final concise summary.'\n"
            "Ensure each item includes name, instruction, step, runMode, dependsOn where applicable."
        ),
    )

manager = build_manager(Path(__file__).parent)


