"""Composition root: the one place where services are constructed and wired.

Everything else receives its dependencies through constructors, so tests (and the
evaluation ablations) can build variants by passing different settings or fakes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from anvil.agents.coding_agent import CodingAgent
from anvil.agents.planner import Planner
from anvil.config import ModelRole, Settings
from anvil.llm.client import ChatModel, Endpoint, FallbackLLM
from anvil.memory.sessions import InMemorySessionStore
from anvil.service import AgentService
from anvil.tools.base import Tool
from anvil.tools.changes import ShowChanges
from anvil.tools.commands import RunCommand
from anvil.tools.filesystem import EditFile, ListFiles, ReadFile, WriteFile
from anvil.tools.registry import ToolRegistry
from anvil.tools.search import FindFiles, SearchCode
from anvil.tools.workspace import WorkspaceManager
from anvil.utils.tracing import TraceSink

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Container:
    settings: Settings
    llms: dict[ModelRole, ChatModel]
    registry: ToolRegistry
    workspaces: WorkspaceManager
    service: AgentService


def build_llm(settings: Settings, role: ModelRole) -> FallbackLLM:
    """A fallback chain for ``role``; providers whose API key is unset are skipped."""
    clients: dict[str, AsyncOpenAI] = {}
    endpoints: list[Endpoint] = []
    for ref in settings.models.get(role, []):
        provider = settings.providers[ref.provider]
        api_key = os.environ.get(provider.api_key_env)
        if not api_key:
            logger.warning(
                "provider skipped: API key not set",
                extra={"provider": ref.provider, "env": provider.api_key_env, "role": role},
            )
            continue
        if ref.provider not in clients:
            clients[ref.provider] = AsyncOpenAI(
                base_url=provider.base_url,
                api_key=api_key,
                timeout=settings.llm.timeout_seconds,
                max_retries=0,  # FallbackLLM owns the retry policy
            )
        endpoints.append(Endpoint(ref.provider, ref.model, clients[ref.provider]))
    return FallbackLLM(
        role,
        endpoints,
        settings.llm.temperature,
        retry_rounds=settings.llm.retry_rounds,
        max_retry_wait=settings.llm.max_retry_wait_seconds,
    )


def build_tools(settings: Settings) -> list[Tool[Any]]:
    """Every available tool, filtered by ``agent.tools`` if set. Add new tools here."""
    tools: list[Tool[Any]] = [
        ListFiles(),
        FindFiles(),
        SearchCode(),
        ReadFile(),
        EditFile(),
        WriteFile(),
        ShowChanges(),
        RunCommand(settings.tools.commands, settings.tools.command_timeout_seconds),
    ]
    enabled = settings.agent.tools
    if enabled is None:
        return tools
    unknown = set(enabled) - {t.name for t in tools}
    if unknown:
        raise ValueError(f"agent.tools: unknown tool(s) {sorted(unknown)}")
    return [t for t in tools if t.name in enabled]


def build_container(
    settings: Settings, llms: dict[ModelRole, ChatModel] | None = None
) -> Container:
    """Wire the application. ``llms`` replaces the real clients (tests use fakes)."""
    if llms is None:
        llms = {role: build_llm(settings, role) for role in ("main", "fast", "judge")}
    registry = ToolRegistry(build_tools(settings))
    workspaces = WorkspaceManager(settings.workspace.template, settings.workspace.root)
    planner = (
        Planner(llms["fast"], registry, settings.agent.prompts.planner)
        if settings.agent.planner
        else None
    )
    agent = CodingAgent(llms["main"], registry, settings.agent, planner)
    sessions = InMemorySessionStore(
        workspaces, settings.memory.max_sessions, settings.memory.ttl_seconds
    )
    service = AgentService(agent, sessions, workspaces, TraceSink(settings.tracing.dir), settings)
    return Container(settings, llms, registry, workspaces, service)
