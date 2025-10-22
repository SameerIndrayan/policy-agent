# agent.py
"""
Policy Agent core logic.

Expose ONE public function:
    evaluate(text: str, policy: str | None) -> tuple[bool, str]

Policy formats supported (pick any one):
1) Plain text, comma-separated blocked words
   Example: "forbidden, secret, ban-this"

2) JSON or YAML-like (YAML without anchors/etc. is OK since we don't parse it with PyYAML):
   {
     "blocked_keywords": ["forbidden", "secret"],
     "blocked_regex": ["(?i)password\\s*[:=]"],
     "max_length": 2000
   }

Rules are ANDed: the first violation blocks and returns the reason.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Tuple


@dataclass
class CompiledPolicy:
    blocked_keywords: list[str] = field(default_factory=list)   # simple substring match (case-insensitive)
    blocked_regex: list[re.Pattern] = field(default_factory=list)
    max_length: int | None = None


def _parse_policy(policy_text: str | None) -> CompiledPolicy:
    """Parse user-provided policy string into a CompiledPolicy."""
    cp = CompiledPolicy()
    if not policy_text or not policy_text.strip():
        return cp

    raw = policy_text.strip()

    # Try JSON first
    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
            _load_structured_policy(obj, cp)
            return cp
        except Exception:
            # fall through to CSV parse
            pass

    # Try very-lightweight YAML (only for a single top-level dict with simple lists/ints).
    # This avoids bringing in PyYAML; if it looks YAML-ish, do a tiny parse.
    if ":" in raw and "\n" in raw:
        maybe = _very_naive_yaml_to_json(raw)
        if maybe is not None:
            try:
                obj = json.loads(maybe)
                _load_structured_policy(obj, cp)
                return cp
            except Exception:
                pass  # fall through

    # Fallback: treat the whole policy as comma-separated blocked keywords
    kws = [w.strip() for w in raw.split(",") if w.strip()]
    cp.blocked_keywords.extend(k.lower() for k in kws)
    return cp


def _load_structured_policy(obj: dict, cp: CompiledPolicy) -> None:
    """Fill CompiledPolicy from a dict object."""
    # blocked_keywords
    bk = obj.get("blocked_keywords") or obj.get("blockedWords") or obj.get("denylist") or []
    if isinstance(bk, Iterable) and not isinstance(bk, (str, bytes)):
        cp.blocked_keywords.extend(str(x).lower() for x in bk if str(x).strip())

    # blocked_regex
    br = obj.get("blocked_regex") or obj.get("blockedRegex") or []
    if isinstance(br, Iterable) and not isinstance(br, (str, bytes)):
        for pattern in br:
            try:
                cp.blocked_regex.append(re.compile(str(pattern)))
            except re.error:
                # ignore invalid patterns
                continue

    # max_length
    ml = obj.get("max_length") or obj.get("maxLength")
    if isinstance(ml, int) and ml > 0:
        cp.max_length = ml


def _very_naive_yaml_to_json(yaml_text: str) -> str | None:
    """
    Extremely naive YAML → JSON for simple cases like:

    blocked_keywords:
      - forbidden
      - secret
    max_length: 2000

    This is intentionally minimal to avoid a PyYAML dependency.
    If it can't confidently convert, return None.
    """
    try:
        lines = [ln.rstrip() for ln in yaml_text.splitlines()]
        indent_stack = [0]
        out = []
        ctx_stack = []  # track whether we're inside list or dict
        # We'll build a JSON-like string by hand (again: only simple cases)

        def open_dict():
            out.append("{"); ctx_stack.append("dict")
        def close_dict():
            out.append("}"); ctx_stack.pop()
        def open_list():
            out.append("["); ctx_stack.append("list")
        def close_list():
            out.append("]"); ctx_stack.pop()

        # Start as dict
        open_dict()
        first_item_written_at_level: dict[int, bool] = {0: False}

        def write_comma_if_needed(level: int):
            if first_item_written_at_level.get(level):
                out.append(",")
            else:
                first_item_written_at_level[level] = True

        level = 0
        key_pending = None

        for raw in lines:
            if not raw.strip():
                continue
            cur_indent = len(raw) - len(raw.lstrip(" "))

            # adjust indentation (step = 2 spaces assumed)
            while cur_indent < indent_stack[-1]:
                # close any open list
                if ctx_stack and ctx_stack[-1] == "list":
                    close_list()
                # close dict if needed
                if ctx_stack and ctx_stack[-1] == "dict":
                    # do nothing; dict closed when parent consumes key?
                    pass
                indent_stack.pop()
                level -= 1
            if cur_indent > indent_stack[-1]:
                indent_stack.append(cur_indent)
                level += 1
                first_item_written_at_level[level] = False

            s = raw.strip()
            if s.startswith("- "):  # list item
                item = s[2:].strip()
                if not (ctx_stack and ctx_stack[-1] == "list"):
                    # open list after a key:
                    out.append("[")
                    ctx_stack.append("list")
                else:
                    out.append(",")
                out.append(json.dumps(item))
                continue

            if ":" in s:
                key, val = s.split(":", 1)
                key = key.strip()
                val = val.strip()

                # if previous was an open list, close it
                if ctx_stack and ctx_stack[-1] == "list":
                    close_list()

                write_comma_if_needed(level - 1 if level > 0 else 0)
                out.append(json.dumps(key))
                out.append(":")
                if val == "":
                    # start nested dict or list; assume dict until we see '- ' on next lines
                    open_dict()
                    first_item_written_at_level[level] = False
                else:
                    # scalar value
                    if val.isdigit():
                        out.append(val)
                    else:
                        out.append(json.dumps(val))
            else:
                # unknown shape → give up
                return None

        # close any open list
        if ctx_stack and ctx_stack[-1] == "list":
            close_list()
        # close root dict
        close_dict()
        return "".join(out)
    except Exception:
        return None


def _check_length(text: str, max_len: int | None) -> Tuple[bool, str | None]:
    if max_len is not None and len(text) > max_len:
        return False, f"Text length {len(text)} exceeds max_length {max_len}."
    return True, None


def _check_blocked_keywords(text: str, keywords: Iterable[str]) -> Tuple[bool, str | None]:
    low = text.lower()
    for kw in keywords:
        if kw and kw in low:
            return False, f"Blocked by keyword: '{kw}'."
    return True, None


def _check_blocked_regex(text: str, patterns: Iterable[re.Pattern]) -> Tuple[bool, str | None]:
    for rx in patterns:
        if rx.search(text):
            return False, f"Blocked by regex: /{rx.pattern}/."
    return True, None


def evaluate(text: str, policy: str | None = None) -> tuple[bool, str]:
    """
    Main entrypoint.
    Returns (allowed, reason). 'reason' explains the first violation or success note.
    """
    if not (text or "").strip():
        return False, "No text provided."

    cp = _parse_policy(policy)

    ok, why = _check_length(text, cp.max_length)
    if not ok:
        return False, why  # length violation

    ok, why = _check_blocked_keywords(text, cp.blocked_keywords)
    if not ok:
        return False, why  # keyword violation

    ok, why = _check_blocked_regex(text, cp.blocked_regex)
    if not ok:
        return False, why  # regex violation

    # Default sample rule (remove if you don't want any built-in deny word):
    if "forbidden" in text.lower():
        return False, "Contains a default forbidden keyword."

    return True, "No issues found."
