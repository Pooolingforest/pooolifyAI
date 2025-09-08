from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


TEMPLATE_RELATIVE = Path("examples/00_simple_agent")


def _copy_template(dst_dir: Path) -> None:
    src = Path(__file__).resolve().parent.parent / "examples" / "00_simple_agent"
    if not src.exists():
        # Fallback: when installed as a package, we also ship under pooolify/examples
        src = Path(__file__).resolve().parent / "examples" / "00_simple_agent"
    if not src.exists():
        raise RuntimeError(f"Template not found at {src}")

    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise FileExistsError(f"Destination directory '{dst_dir}' is not empty")

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Copy tree
    shutil.copytree(src, dst_dir, dirs_exist_ok=True)


def cmd_create_new(args: argparse.Namespace) -> int:
    project_name: str = args.project_name
    target: Path = Path(args.path).expanduser().resolve() if args.path else Path.cwd() / project_name
    try:
        _copy_template(target)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Created pooolify project at: {target}")
    print("Next steps:")
    print(f"  1) cd {target}")
    print("  2) Create and edit .env (set LLM_OPENAI_API_KEY)")
    print("  3) Run: uv sync && uv run uvicorn main:app --reload")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pooolify", description="pooolify CLI")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="Create resources")
    create_sub = create_parser.add_subparsers(dest="create_command")

    new_parser = create_sub.add_parser("new", help="Scaffold a new project from 00_simple_agent")
    new_parser.add_argument("project_name", help="Name of the project directory to create")
    new_parser.add_argument("--path", help="Optional path where the project should be created (defaults to CWD/project_name)")
    new_parser.set_defaults(func=cmd_create_new)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if not hasattr(ns, "func"):
        parser.print_help()
        return 0
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())


