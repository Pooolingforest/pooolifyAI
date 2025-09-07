from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pooolify.tools.base import Tool


class UtcTimeTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="utc_time", description="Return current UTC time in ISO and unix")

    def run(self, *, fmt: Optional[str] = None) -> Dict[str, Any]:  # noqa: ARG002
        now = datetime.now(timezone.utc)
        return {
            "timezone": "UTC",
            "now_iso": now.isoformat(),
            "now_unix": int(now.timestamp()),
        }


def build_tool(base_dir):  # noqa: ANN001, ANN201, ARG001
    return UtcTimeTool()


