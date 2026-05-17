"""Defense monitor: runs perplexity baselines and per-problem post-TTT passes."""

from __future__ import annotations

import logging
import math
from typing import Any

from hydra.utils import instantiate

from .perplexity import (
    compute_dataset_perplexity,
    generate_clean_targets,
    save_defense_results,
)

log = logging.getLogger(__name__)


class DefenseMonitor:
    """Wraps a TTT backend with perplexity-based defense state.

    Lifecycle: construct → :meth:`baseline` (once, before the loop) →
    :meth:`after_fit` (per problem, after the backend's ``fit`` call). The
    monitor holds the loaded harmful/clean holdouts, baseline perplexities,
    and any generated clean targets so the per-problem call is just one
    forward pass against the (now-trained) backend.

    Requires the backend to advertise ``supports_perplexity = True`` and to
    expose ``.model`` and ``.tokenizer`` attributes (the monitor reads them
    directly to drive batched perplexity and clean-target generation).
    """

    def __init__(self, backend, defense_cfg: Any, data_cfg: Any):
        if not backend.supports_perplexity:
            raise RuntimeError(
                f"Defense (perplexity monitoring) requested but backend "
                f"{type(backend).__name__} does not support perplexity()."
            )
        for attr in ("model", "tokenizer"):
            if not hasattr(backend, attr):
                raise RuntimeError(
                    f"DefenseMonitor requires backend.{attr}; "
                    f"{type(backend).__name__} does not expose it."
                )
        self.backend = backend
        self.cfg = defense_cfg
        self.data_cfg = data_cfg

        self.harmful_holdout = None
        self.clean_holdout = None
        self.clean_targets = None
        self.clean_target_column = defense_cfg.get("clean_target_column")
        self.clean_has_pretargets = False

        self.harmful_before = None
        self.clean_before = None

    def baseline(self) -> None:
        """Load holdouts and compute pre-TTT perplexity baselines (called once)."""
        cfg = self.cfg
        if cfg.get("harmful_holdout"):
            log.info("Defense: loading harmful holdout dataset...")
            self.harmful_holdout = instantiate(
                cfg.harmful_holdout, tokenizer=self.backend.tokenizer
            )
            log.info(f"Harmful holdout loaded: {len(self.harmful_holdout)} examples")

        if cfg.get("clean_holdout"):
            log.info("Defense: loading clean holdout dataset...")
            try:
                self.clean_holdout = instantiate(
                    cfg.clean_holdout, tokenizer=self.backend.tokenizer
                )
            except TypeError:
                self.clean_holdout = instantiate(cfg.clean_holdout)
            log.info(f"Clean holdout loaded: {len(self.clean_holdout)} examples")

        self.clean_has_pretargets = (
            self.clean_holdout is not None
            and self.clean_target_column is not None
            and self.clean_target_column in self.clean_holdout.column_names
        )

        if self.harmful_holdout is not None:
            log.info("Defense: harmful baseline perplexity (before TTT)...")
            self.harmful_before = self._harmful_perplexity()

        if self.clean_holdout is not None:
            if not self.clean_has_pretargets:
                log.info("Defense: generating clean targets from base model...")
                self.clean_targets = generate_clean_targets(
                    model=self.backend.model,
                    tokenizer=self.backend.tokenizer,
                    dataset=self.clean_holdout,
                    goal_column=cfg.get("clean_goal_column", "question"),
                    system_prompt=self.data_cfg.get(
                        "system_prompt", "You are a helpful assistant."
                    ),
                    max_new_tokens=cfg.get("clean_max_new_tokens", 25),
                    min_new_tokens=cfg.get("clean_min_new_tokens", 5),
                    temperature=cfg.get("clean_temperature", 1.0),
                    batch_size=cfg.get("batch_size", 8),
                    enable_thinking=self.data_cfg.get("enable_thinking", None),
                )
                log.info(f"Generated {len(self.clean_targets)} clean targets")

            log.info("Defense: clean baseline perplexity (before TTT)...")
            self.clean_before = self._clean_perplexity()

    def after_fit(self, global_idx: int, output_dir: str) -> None:
        """Compute post-TTT perplexity for ``global_idx`` and persist the row."""
        harmful_after = None
        clean_after = None

        if self.harmful_holdout is not None and self.harmful_before is not None:
            log.info(f"Defense: harmful perplexity after TTT (problem {global_idx})...")
            harmful_after = self._harmful_perplexity()
            self._log_shift("harmful", self.harmful_before, harmful_after)

        if self.clean_holdout is not None and self.clean_before is not None:
            log.info(f"Defense: clean perplexity after TTT (problem {global_idx})...")
            clean_after = self._clean_perplexity()
            self._log_shift("clean", self.clean_before, clean_after)

        if self.harmful_before is not None and harmful_after is not None:
            save_defense_results(
                self.harmful_before,
                harmful_after,
                output_dir,
                global_idx,
                clean_ppl_before=self.clean_before,
                clean_ppl_after=clean_after,
            )

    def _harmful_perplexity(self):
        cfg = self.cfg
        hh = self.harmful_holdout
        return compute_dataset_perplexity(
            model=self.backend.model,
            tokenizer=self.backend.tokenizer,
            dataset=hh,
            batch_size=cfg.get("batch_size", 8),
            max_length=cfg.get("max_length"),
            prompt_column="prompt" if "prompt" in hh.column_names else None,
            goal_column=cfg.get("harmful_goal_column", "goal"),
            target_column=cfg.get("harmful_target_column", "target"),
            prefix_lengths=cfg.get("prefix_lengths"),
            enable_thinking=self.data_cfg.get("enable_thinking", None),
        )

    def _clean_perplexity(self):
        cfg = self.cfg
        ch = self.clean_holdout
        prompt_column = (
            "prompt"
            if self.clean_has_pretargets and "prompt" in ch.column_names
            else None
        )
        return compute_dataset_perplexity(
            model=self.backend.model,
            tokenizer=self.backend.tokenizer,
            dataset=ch,
            batch_size=cfg.get("batch_size", 8),
            max_length=cfg.get("max_length"),
            prompt_column=prompt_column,
            goal_column=cfg.get("clean_goal_column", "question"),
            target_column=self.clean_target_column or "target",
            targets=None if self.clean_has_pretargets else self.clean_targets,
            prefix_lengths=cfg.get("prefix_lengths"),
            enable_thinking=self.data_cfg.get("enable_thinking", None),
        )

    @staticmethod
    def _log_shift(label: str, before, after) -> None:
        # NaN-safe: rows with empty target_mask emit NaN per
        # defense.perplexity._per_batch_perplexity; nanmean keeps the headline
        # reading useful instead of going NaN on a single truncated row.
        mean_b = _nanmean(before["perplexity_avg"])
        mean_a = _nanmean(after["perplexity_avg"])
        log.info(f"Defense: {label} PPL before={mean_b:.2f}, after={mean_a:.2f}")


def _nanmean(values: list[float]) -> float:
    real = [v for v in values if not math.isnan(v)]
    return sum(real) / len(real) if real else float("nan")
