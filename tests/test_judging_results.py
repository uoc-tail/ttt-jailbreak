"""Tests for judging/results.py: legacy-shape coercion and ASR aggregation."""

from __future__ import annotations

from collections import Counter

from ttt_jailbreak.judging import compute_stage_stats, normalize_generations


class TestNormalizeGenerations:
    def test_list_wrapped_under_default(self):
        data = [{"goal": "g", "generations": ["a", "b"]}]
        normalize_generations(data)
        assert data[0]["generations"] == {"default": ["a", "b"]}

    def test_dict_with_bare_string_values_wrapped_in_lists(self):
        data = [{"goal": "g", "generations": {"stage1": "single"}}]
        normalize_generations(data)
        assert data[0]["generations"] == {"stage1": ["single"]}

    def test_dict_of_lists_untouched(self):
        data = [{"goal": "g", "generations": {"stage1": ["a", "b"]}}]
        normalize_generations(data)
        assert data[0]["generations"] == {"stage1": ["a", "b"]}

    def test_legacy_rs_sequential_reconstructs(self):
        # Legacy RS rows had flat `before_generation` / `final_generation` /
        # `ttt_mode` fields. normalize_generations must rebuild the dict shape.
        data = [
            {
                "goal": "g",
                "prompt_template": "refined_best",
                "before_generation": "X",
                "final_generation": "Y",
                "before_ttt_generation": "Z",
                "ttt_mode": "sequential",
            }
        ]
        normalize_generations(data)
        gens = data[0]["generations"]
        assert gens == {
            "manual_prompt": ["X"],
            "rs": ["Z"],
            "rs+ttt": ["Y"],
        }

    def test_legacy_rs_simple_template_uses_no_attack(self):
        data = [
            {
                "goal": "g",
                "prompt_template": "simple",
                "before_generation": "X",
                "final_generation": "Y",
            }
        ]
        normalize_generations(data)
        assert "no_attack" in data[0]["generations"]
        assert data[0]["generations"]["no_attack"] == ["X"]


class TestComputeStageStats:
    def _record(self, jb, of, reason=None):
        n = len(jb)
        return {
            "is_jailbroken": {"s": jb},
            "is_overfitted": {"s": of},
            "invalid_reason": {"s": reason or [""] * n},
        }

    def test_empty_data_returns_zeros(self):
        stats = compute_stage_stats([], "s")
        assert stats["total"] == 0
        assert stats["n_problems"] == 0
        assert stats["k"] == 0

    def test_basic_counts(self):
        data = [
            self._record([True, False], [False, False]),
            self._record([True, True], [False, False]),
        ]
        stats = compute_stage_stats(data, "s")
        assert stats["total"] == 4
        assert stats["jailbroken"] == 3
        assert stats["n_problems"] == 2
        assert stats["k"] == 2
        assert stats["prompts_any"] == 2

    def test_valid_jb_excludes_overfitted(self):
        data = [
            self._record([True, True], [True, False]),
        ]
        stats = compute_stage_stats(data, "s")
        assert stats["jailbroken"] == 2
        assert stats["invalid"] == 1
        assert stats["valid_total"] == 1
        assert stats["valid_jb"] == 1

    def test_reason_counts_collected(self):
        data = [
            self._record(
                [False, False],
                [True, True],
                reason=["too_short", "too_short"],
            )
        ]
        stats = compute_stage_stats(data, "s")
        assert stats["reason_counts"] == Counter({"too_short": 2})

    def test_missing_stage_skipped(self):
        data = [{"is_jailbroken": {}, "is_overfitted": {}, "invalid_reason": {}}]
        stats = compute_stage_stats(data, "s")
        assert stats["n_problems"] == 0
