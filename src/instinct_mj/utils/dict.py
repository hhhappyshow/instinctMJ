"""Dictionary utilities."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from typing import Any


def class_to_dict(obj: Any) -> Any:
    """Convert a config object into plain data recursively.

  Mirrors ``isaaclab.utils.dict.class_to_dict``, which IsaacLab's ``dump_yaml``
  applied for us. mjlab's ``dump_yaml`` is a bare ``yaml.dump``, so configs have
  to be reduced here instead.

  ``dataclasses.asdict`` is not enough: it leaves class and function objects in
  the tree, so ``yaml.dump`` falls back to ``!!python/name`` and
  ``!!python/object`` tags. The dump still succeeds, but the result can no
  longer be read back with ``yaml.safe_load``. Config trees here are full of
  such objects -- every ``class_type`` and every manager term ``func`` -- so
  those are rendered as ``"module:qualname"`` strings instead. A
  ``functools.partial`` keeps its function plus the bound arguments.
  """
    if isinstance(obj, Enum):
        return class_to_dict(obj.value)
    if is_dataclass(obj):
        return {item.name: class_to_dict(getattr(obj, item.name)) for item in fields(obj)}
    if isinstance(obj, dict):
        return {str(key): class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return [class_to_dict(value) for value in obj]
    if isinstance(obj, list):
        return [class_to_dict(value) for value in obj]
    if isinstance(obj, partial):
        return {
            "func": class_to_dict(obj.func),
            "args": class_to_dict(obj.args),
            "keywords": class_to_dict(obj.keywords),
        }
    if callable(obj):
        return f"{obj.__module__}:{obj.__qualname__}"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)
