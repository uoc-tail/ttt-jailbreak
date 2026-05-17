"""Perplexity-based defense against TTT jailbreaks."""

from .monitor import DefenseMonitor
from .perplexity import (
    compute_dataset_perplexity,
    generate_clean_targets,
    save_defense_results,
)

__all__ = [
    "DefenseMonitor",
    "compute_dataset_perplexity",
    "generate_clean_targets",
    "save_defense_results",
]
