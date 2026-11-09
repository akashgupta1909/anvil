"""Command-line entry point: ``run``, ``chat``, ``serve`` and ``eval`` subcommands."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from anvil.config import Settings, load_settings
from anvil.container import build_container
from anvil.models import (
    AgentEvent,
    AgentResult,
    FinalEvent,
    PlanEvent,
    RunStatus,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from anvil.service import AgentService
from anvil.tools.approval import AllowAll
from anvil.tools.base import Approver
from anvil.tools.workspace import Workspace
from anvil.utils.logging import configure_logging

EXIT_CODES = {RunStatus.COMPLETED: 0, RunStatus.ERROR: 1, RunStatus.MAX_STEPS: 2}


class PromptApprover:
    """Asks on the terminal before a side-effecting tool runs; denies without a TTY."""

    async def approve(self, tool: str, arguments: dict[str, Any]) -> bool:
        if not sys.stdin.isatty():
            return False
        preview = json.dumps(arguments)
        if len(preview) > 300:
            preview = preview[:300] + "…"
        answer = await asyncio.to_thread(input, f"\nAllow {tool} {preview}? [y/N] ")
        return answer.strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except (OSError, ValidationError) as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging("INFO" if args.command == "serve" else "WARNING")

    if args.command == "serve":
        return _serve(settings, args.host, args.port)
    if args.command == "eval":
        from anvil.evaluation.runner import run_evaluation

        return asyncio.run(
            run_evaluation(
                settings,
                ablations=args.ablation,
                task_ids=args.task,
                repeats=args.repeats,
                use_judge=not args.no_judge,
            )
        )

    service = build_container(settings).service
    approver: Approver = AllowAll() if args.yes else PromptApprover()
    workspace = Workspace(args.workspace) if args.workspace else None
    if args.command == "chat":
        return asyncio.run(_chat(service, approver, workspace, args.verbose))
    return asyncio.run(_run(service, args.task, approver, workspace, args.verbose, args.json))


async def _run(
    service: AgentService,
    task: str,
    approver: Approver,
    workspace: Workspace | None,
    verbose: bool,
    as_json: bool,
) -> int:
    result: AgentResult | None = None
    async for event in service.run_stream(task, approver, workspace=workspace):
        if isinstance(event, FinalEvent):
            result = event.result
        elif not as_json:
            _print_event(event, verbose)
    assert result is not None
    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        _print_summary(result)
    return EXIT_CODES[result.status]


async def _chat(
    service: AgentService, approver: Approver, workspace: Workspace | None, verbose: bool
) -> int:
    session = service.sessions.create(workspace)
    print(f"Workspace: {session.workspace.root}\nType a task, or 'exit' to quit.", file=sys.stderr)
    while True:
        try:
            task = (await asyncio.to_thread(input, "\n> ")).strip()
        except EOFError:
            break
        if task in {"exit", "quit"}:
            break
        if not task:
            continue
        async for event in service.run_stream(task, approver, session_id=session.id):
            if isinstance(event, FinalEvent):
                _print_summary(event.result)
            else:
                _print_event(event, verbose)
    service.sessions.delete(session.id)
    return 0


def _print_event(event: AgentEvent, verbose: bool) -> None:
    if isinstance(event, TextDeltaEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, PlanEvent):
        print("Plan:\n" + "\n".join(f"  - {s}" for s in event.steps), file=sys.stderr)
    elif isinstance(event, ToolCallEvent):
        args = event.arguments if verbose else event.arguments[:120]
        print(f"[{event.step}] → {event.name} {args}", file=sys.stderr)
    elif isinstance(event, ToolResultEvent):
        mark = "✓" if event.ok else "✗"
        print(f"    {mark} {event.name} ({event.duration_ms:.0f} ms)", file=sys.stderr)
        if verbose or not event.ok:
            print("      " + event.output.replace("\n", "\n      "), file=sys.stderr)


def _print_summary(result: AgentResult) -> None:
    print(file=sys.stderr)
    if result.status is not RunStatus.COMPLETED:
        print(f"[{result.status.value}] {result.answer}", file=sys.stderr)
    print(
        f"-- {result.status.value} · {result.steps} steps · {len(result.tool_calls)} tool calls · "
        f"{result.usage.input_tokens}+{result.usage.output_tokens} tokens · "
        f"{result.duration_ms / 1000:.1f}s · trace {result.request_id}",
        file=sys.stderr,
    )


def _serve(settings: Settings, host: str, port: int) -> int:
    import uvicorn

    from anvil.api.app import create_app

    configure_logging(settings.logging.level)
    uvicorn.run(create_app(build_container(settings)), host=host, port=port, log_config=None)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anvil", description=__doc__)
    parser.add_argument("--config", type=Path, help="config file (default: config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_agent_options(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--workspace",
            type=Path,
            help="directory the agent works in (default: a disposable copy of the template)",
        )
        p.add_argument(
            "-y", "--yes", action="store_true", help="approve file writes and code execution"
        )
        p.add_argument("-v", "--verbose", action="store_true", help="show full tool output")

    run = sub.add_parser("run", help="run one task")
    run.add_argument("task")
    run.add_argument("--json", action="store_true", help="print the result as JSON")
    add_agent_options(run)

    add_agent_options(sub.add_parser("chat", help="interactive session with memory"))

    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    ev = sub.add_parser("eval", help="run the evaluation suite")
    ev.add_argument(
        "--ablation", action="append", help="variant(s) to run (repeatable; default: baseline)"
    )
    ev.add_argument("--task", action="append", help="only these task ids (repeatable)")
    ev.add_argument("--repeats", type=int, default=1)
    ev.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    return parser
