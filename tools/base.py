from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Tool:
    name: str
    description: str
    timeout_s: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

    def run(self, **kwargs: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def safe_call(self, **kwargs: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            output = self.run(**kwargs)
            ok = True
        except Exception as e:  # noqa: BLE001
            output = {"error": str(e)}
            ok = False
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {"success": ok, "latency_ms": latency_ms, "output": output}

