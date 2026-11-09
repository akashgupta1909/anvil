"""Versioned prompt templates.

Templates live in ``templates/<name>.<version>.md`` and use ``$placeholders``
(``string.Template``). A new prompt version is a new file, so old versions stay
reproducible, and every rendered prompt carries its version and content hash for traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from string import Template

from anvil.utils.hashing import sha256_text


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def ref(self) -> str:
        """``name@version#hash`` — identifies the exact text in traces."""
        return f"{self.name}@{self.version}#{sha256_text(self.text)[:12]}"


@cache
def _template(name: str, version: str) -> str:
    path = resources.files(__package__) / "templates" / f"{name}.{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt template {name!r} version {version!r}")
    return path.read_text()


def render(name: str, version: str, **values: str) -> Prompt:
    return Prompt(name, version, Template(_template(name, version)).substitute(values).strip())


def wrap_data(kind: str, content: str, **attrs: str) -> str:
    """Mark untrusted content (tool output, file text, agent answers) as data, not
    instructions. This is prompt hygiene, not a security boundary: the real controls
    are the workspace sandbox and the approval gate for side-effecting tools."""
    attr_text = "".join(f' {k}="{v}"' for k, v in attrs.items())
    # Neutralise a closing tag inside the content so it cannot end the block early.
    safe = content.replace(f"</{kind}>", f"<\\/{kind}>")
    return f"<{kind}{attr_text}>\n{safe}\n</{kind}>"
