"""TTT (test-time training) backend API."""

from .base import FinetuneMode, TTTBackend
from .huggingface import HuggingFaceBackend


def __getattr__(name):
    # Lazy import TinkerBackend so HF-only users do not pay for tinker imports.
    if name == "TinkerBackend":
        from .tinker import TinkerBackend

        return TinkerBackend
    raise AttributeError(name)


__all__ = ["FinetuneMode", "TTTBackend", "HuggingFaceBackend", "TinkerBackend"]
