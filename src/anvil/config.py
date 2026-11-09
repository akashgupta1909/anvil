"""Typed application settings.

Non-secret settings come from ``config.yaml`` and can be overridden by environment
variables named ``ANVIL__<SECTION>__<KEY>``. Secrets (API keys) are never stored
in settings: providers name the env var that holds their key, and it is read from the
environment (optionally populated from ``.env``) when a client is built.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

ModelRole = Literal["main", "fast", "judge"]
API_KEY_ENV = "ANVIL_API_KEY"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderSettings(_Section):
    base_url: str
    api_key_env: str


class ModelRef(_Section):
    provider: str
    model: str


class LLMSettings(_Section):
    timeout_seconds: float = Field(default=60, gt=0)
    retry_rounds: int = Field(default=2, ge=0)
    max_retry_wait_seconds: float = Field(default=30, ge=0)
    temperature: float = Field(default=0, ge=0, le=2)


class Price(_Section):
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class PromptVersions(_Section):
    system: str = "v1"
    planner: str = "v1"
    judge: str = "v1"


class AgentSettings(_Section):
    max_steps: PositiveInt = 12
    planner: bool = False
    wrap_tool_output: bool = True
    prompts: PromptVersions = PromptVersions()


class ToolSettings(_Section):
    max_output_chars: PositiveInt = 10_000
    run_timeout_seconds: float = Field(default=30, gt=0)


class WorkspaceSettings(_Section):
    template: Path = Path("examples/calculator")
    root: Path = Path(".workspaces")


class MemorySettings(_Section):
    max_turns: PositiveInt = 10
    max_sessions: PositiveInt = 100
    ttl_seconds: PositiveInt = 3600


class ApiSettings(_Section):
    require_auth: bool = True


class TracingSettings(_Section):
    dir: Path = Path("logs/traces")


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANVIL__",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    providers: dict[str, ProviderSettings]
    models: dict[ModelRole, list[ModelRef]]
    llm: LLMSettings = LLMSettings()
    pricing: dict[str, Price] = {}
    agent: AgentSettings = AgentSettings()
    tools: ToolSettings = ToolSettings()
    workspace: WorkspaceSettings = WorkspaceSettings()
    memory: MemorySettings = MemorySettings()
    api: ApiSettings = ApiSettings()
    tracing: TracingSettings = TracingSettings()
    logging: LoggingSettings = LoggingSettings()

    @model_validator(mode="after")
    def _check_model_refs(self) -> Settings:
        if not self.models.get("main"):
            raise ValueError("models.main must list at least one provider/model")
        for role, chain in self.models.items():
            for ref in chain:
                if ref.provider not in self.providers:
                    raise ValueError(f"models.{role}: unknown provider {ref.provider!r}")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Env vars override the YAML values that are passed in as init kwargs.
        return env_settings, init_settings


def load_settings(path: Path | str | None = None, **overrides: Any) -> Settings:
    """Load ``.env`` into the environment, then build validated settings from YAML.

    ``overrides`` are deep-merged over the YAML (used by evaluation ablations).
    """
    load_dotenv(override=False)
    config_path = Path(path or os.environ.get("ANVIL_CONFIG", "config.yaml"))
    data: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    return Settings(**deep_merge(data, overrides))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            merged[key] = value
    return merged
