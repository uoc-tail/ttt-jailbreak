"""Few-shot example sampling + per-problem (goal, target) extraction."""

from __future__ import annotations

import logging
import random
from typing import Optional

from datasets import Dataset

log = logging.getLogger(__name__)


def sample_few_shot_examples(
    full_dataset: Dataset,
    target_idx: int,
    num_shots: int,
    seed: Optional[int] = None,
) -> list[int]:
    """Sample ``num_shots`` dataset indices, excluding ``target_idx``.

    With ``seed=None`` the private RNG is OS-seeded, so the output varies
    across runs. Pass an explicit ``seed`` for reproducibility.
    """
    all_indices = list(range(len(full_dataset)))
    if target_idx in all_indices:
        all_indices.remove(target_idx)

    if len(all_indices) < num_shots:
        log.warning(
            f"Requested {num_shots} shots but only {len(all_indices)} examples "
            f"available (excluding target). Using all available examples."
        )
        return all_indices

    # Always use a private RNG so callers without a seed do not perturb the
    # global random state (which would couple unrelated stochastic code).
    rng = random.Random(seed)
    return rng.sample(all_indices, num_shots)


def example_from_problem(problem: dict) -> tuple[str, str]:
    """Extract the ``(goal, target)`` pair to feed the TTT backend.

    Prefers ``user_content`` over ``goal`` so any dataset-level wrapping
    (universal manual prompts, suffixes) is preserved.
    """
    goal = problem.get("user_content", problem.get("goal", problem.get("Goal", "")))
    target = problem.get("target", problem.get("Target", ""))
    return goal, target
