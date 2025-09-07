from __future__ import annotations

from typing import Any, Dict

from pooolify.tools.base import Tool


class EchoTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="echo", description="Echo back provided input as text")

    def run(self, *, text: str) -> Dict[str, Any]:
        return {"echo": text}


def build_tool(base_dir):  # noqa: ANN001, ANN201, ARG001
    return EchoTool()


