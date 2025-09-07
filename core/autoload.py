from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, List

from ..agents.runtime import AgentDefinition
from ..tools.base import Tool
from ..core.app import PooolifyApp


def _import_from_file(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:  # pragma: no cover - safety
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_tools(base_dir: Path, *, dir_name: str = "tools") -> Dict[str, Tool]:
    """
    Load tools from base_dir/dir_name/<tool_name>/index.py.
    Convention per tool module:
      - Prefer a callable `build_tool(base_dir: Path) -> Tool`
      - Else a variable `tool: Tool`
    The returned dict is keyed by tool.name if available; otherwise by folder name.
    """
    tools: Dict[str, Tool] = {}
    tools_root = base_dir / dir_name
    if not tools_root.exists():
        return tools

    for child in tools_root.iterdir():
        if not child.is_dir():
            continue
        index_file = child / "index.py"
        if not index_file.exists():
            continue
        module_name = f"pooolify_autoload.tools.{child.name}"
        mod = _import_from_file(module_name, index_file)
        instance: Optional[Tool] = None
        build = getattr(mod, "build_tool", None)
        if callable(build):
            instance = build(base_dir)
        if instance is None:
            instance = getattr(mod, "tool", None)
        if not isinstance(instance, Tool):
            raise TypeError(f"Module {index_file} did not produce a Tool instance")
        key = instance.name or child.name
        tools[key] = instance
    return tools


def load_agents(base_dir: Path, *, dir_name: str = "agent", tools: Optional[Dict[str, Tool]] = None) -> Dict[str, AgentDefinition]:
    """
    Load agents from base_dir/dir_name/<agent_name>/index.py.
    Convention per agent module:
      - Prefer `build_agent(tools: Dict[str, Tool], base_dir: Path) -> AgentDefinition`
      - Else a variable `agent: AgentDefinition`
    The returned dict is keyed by agent.name if available; otherwise by folder name.
    """
    agents: Dict[str, AgentDefinition] = {}
    tools = tools or {}
    agents_root = base_dir / dir_name
    if not agents_root.exists():
        return agents

    def _read_docs_markdown(docs_dir: Path) -> str:
        """Read and concatenate markdown files within docs_dir recursively.

        Supported extensions: .md, .mdx, .markdown. Files are sorted for determinism.
        """
        if not docs_dir.exists() or not docs_dir.is_dir():
            return ""
        exts = {".md", ".mdx", ".markdown"}
        files: List[Path] = []
        for p in docs_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
        files.sort(key=lambda p: str(p).lower())
        chunks: List[str] = []
        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8")
            except Exception:
                try:
                    content = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    content = ""
            content = content.strip()
            if content:
                chunks.append(content)
        return "\n\n".join(chunks).strip()

    for child in agents_root.iterdir():
        if not child.is_dir():
            continue
        index_file = child / "index.py"
        if not index_file.exists():
            continue
        module_name = f"pooolify_autoload.agent.{child.name}"
        mod = _import_from_file(module_name, index_file)
        instance: Optional[AgentDefinition] = None
        build = getattr(mod, "build_agent", None)
        if callable(build):
            instance = build(tools, base_dir)
        if instance is None:
            instance = getattr(mod, "agent", None)
        if not isinstance(instance, AgentDefinition):
            raise TypeError(f"Module {index_file} did not produce an AgentDefinition")
        # Default tool injection: if tools not specified, grant access to all loaded tools
        if instance.tools is None:
            instance.tools = list((tools or {}).values())

        # Merge docs/ markdown into knowledge if present
        docs_dir = child / "docs"
        docs_text = _read_docs_markdown(docs_dir)
        if docs_text:
            existing = (instance.knowledge or "").strip()
            if existing:
                instance.knowledge = f"{existing}\n\n{docs_text}"
            else:
                instance.knowledge = docs_text
        key = instance.name or child.name
        agents[key] = instance
    return agents


def load_manager(base_dir: Path, *, dir_name: str = "agent") -> Optional[PooolifyApp.ManagerConfig]:
    """
    Load manager configuration from base_dir/dir_name/index.py.
    Convention per module:
      - Prefer `build_manager(base_dir: Path) -> PooolifyApp.ManagerConfig`
      - Else a variable `manager: PooolifyApp.ManagerConfig | dict`
    If a dict is provided, it's expanded into ManagerConfig.
    """
    index_file = base_dir / dir_name / "index.py"
    if not index_file.exists():
        return None
    module_name = "pooolify_autoload.manager"
    mod = _import_from_file(module_name, index_file)
    instance = None
    build = getattr(mod, "build_manager", None)
    if callable(build):
        instance = build(base_dir)
    if instance is None:
        instance = getattr(mod, "manager", None)
    if instance is None:
        raise TypeError(f"Module {index_file} did not define build_manager() or manager")
    if isinstance(instance, dict):
        return PooolifyApp.ManagerConfig(**instance)
    if isinstance(instance, PooolifyApp.ManagerConfig):
        return instance
    raise TypeError(f"Module {index_file} produced unsupported manager type: {type(instance)}")


