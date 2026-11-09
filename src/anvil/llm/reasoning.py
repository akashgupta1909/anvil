"""Strip ``<think>…</think>`` reasoning blocks that some open models emit inline."""

from __future__ import annotations

import re

_OPEN, _CLOSE = "<think>", "</think>"
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def strip_reasoning(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class ReasoningFilter:
    """Incremental version of :func:`strip_reasoning` for streamed text, where a tag
    can be split across chunks. Text that might be the start of a tag is held back."""

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            if self._inside:
                end = self._buf.find(_CLOSE)
                if end < 0:
                    self._buf = self._buf[-(len(_CLOSE) - 1) :]
                    break
                self._buf = self._buf[end + len(_CLOSE) :]
                self._inside = False
            else:
                start = self._buf.find(_OPEN)
                if start < 0:
                    keep = _partial_prefix_len(self._buf, _OPEN)
                    out.append(self._buf[: len(self._buf) - keep])
                    self._buf = self._buf[len(self._buf) - keep :]
                    break
                out.append(self._buf[:start])
                self._buf = self._buf[start + len(_OPEN) :]
                self._inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buf
        self._buf = ""
        return rest


def _partial_prefix_len(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0
