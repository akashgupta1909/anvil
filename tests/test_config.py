from pathlib import Path

import pytest
from pydantic import ValidationError

from anvil.config import Settings, deep_merge, load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_is_valid() -> None:
    settings = load_settings(ROOT / "config.yaml")
    assert settings.models["main"][0].provider in settings.providers
    assert settings.agent.max_steps > 0


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANVIL__AGENT__MAX_STEPS", "3")
    settings = load_settings(ROOT / "config.yaml")
    assert settings.agent.max_steps == 3
    assert settings.agent.wrap_tool_output is True  # sibling keys keep their YAML values


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown provider"):
        Settings.model_validate(
            {"providers": {}, "models": {"main": [{"provider": "nope", "model": "x"}]}}
        )


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "providers": {"p": {"base_url": "u", "api_key_env": "K"}},
                "models": {"main": [{"provider": "p", "model": "m"}]},
                "agent": {"max_stepz": 3},
            }
        )


def test_deep_merge_replaces_lists_and_merges_dicts() -> None:
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    assert deep_merge(base, {"a": {"y": [3]}}) == {"a": {"x": 1, "y": [3]}, "b": 1}
