"""Benchmark infrastructure for ResearchHelperAgent."""

from .common import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]


def __getattr__(name: str):
    if name in {"LoCoMoAdapter", "LoCoMoEvaluator", "LoCoMoLoader", "LoCoMoRunner", "LoCoMoEnvironment"}:
        from . import locomo

        value = getattr(locomo, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
