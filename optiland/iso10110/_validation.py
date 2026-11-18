"""Minimal stdlib validation helpers for the ISO 10110 spec dataclasses.

Replaces pydantic with (a) a class decorator that rejects unknown keyword
arguments, so typo'd field names fail loudly instead of being silently
ignored, and (b) a light type coercion for numeric strings (e.g. values read
from YAML) based on each field's declared annotation.
"""

from __future__ import annotations

import dataclasses
import functools


def strict_dataclass(cls):
    """Turn *cls* into a dataclass whose ``__init__`` rejects unknown keywords.

    Raises ``ValueError`` (not ``TypeError``) naming the offending and the
    valid fields. Warnings emitted from ``__post_init__`` use ``stacklevel=4``
    so they point at the caller of the constructor.
    """
    cls = dataclasses.dataclass(cls)
    names = tuple(f.name for f in dataclasses.fields(cls))
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        unknown = sorted(set(kwargs) - set(names))
        if unknown:
            raise ValueError(
                f"{cls.__name__}() got unexpected field(s) {unknown} "
                f"(extra fields are not permitted); valid fields: {sorted(names)}"
            )
        orig_init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


def field_names(cls) -> set[str]:
    """Names of all dataclass fields of *cls*."""
    return {f.name for f in dataclasses.fields(cls)}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def coerce_fields(obj) -> None:
    """Validate and coerce every field of dataclass *obj* in place.

    Supported annotations (optionally ``| None``): ``float``, ``int``,
    ``str``, ``bool`` and ``str | float``. Numeric strings are converted for
    ``float``/``int`` fields; anything else of the wrong type raises
    ``ValueError``.
    """
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        parts = {p.strip() for p in str(f.type).split("|")}
        optional = "None" in parts
        parts.discard("None")
        if v is None:
            if not optional:
                raise ValueError(f"{f.name} must not be None")
            continue
        setattr(obj, f.name, _coerce(f.name, v, parts))


def _coerce(name: str, v, parts: set[str]):
    if parts == {"str", "float"}:
        if isinstance(v, str) or _is_number(v):
            return v
    elif parts == {"float"}:
        if _is_number(v):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                pass
    elif parts == {"int"}:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                pass
    elif parts == {"str"}:
        if isinstance(v, str):
            return v
    elif parts == {"bool"}:
        if isinstance(v, bool):
            return v
    else:  # pragma: no cover - guards against unsupported new annotations
        raise TypeError(f"unsupported annotation for {name}: {sorted(parts)}")
    raise ValueError(
        f"{name}: invalid value {v!r} for type {' | '.join(sorted(parts))}"
    )
