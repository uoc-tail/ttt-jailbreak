"""Tests for defense/perplexity.py: defense_results.jsonl write semantics."""

from __future__ import annotations

import json

from ttt_jailbreak.defense.perplexity import save_defense_results


def _ppl(value=1.0):
    return {
        "perplexity_avg": [value],
        "perplexity_first": [value],
        "nll_avg": [0.0],
        "nll_first": [0.0],
    }


def test_writes_one_row_per_problem(tmp_path):
    save_defense_results(_ppl(), _ppl(2.0), str(tmp_path), problem_idx=0)
    save_defense_results(_ppl(), _ppl(2.0), str(tmp_path), problem_idx=1)
    path = tmp_path / "defense" / "defense_results.jsonl"
    rows = [json.loads(line) for line in open(path)]
    assert [r["problem_idx"] for r in rows] == [0, 1]


def test_rerun_does_not_duplicate_rows(tmp_path):
    # Simulate a re-run of the same problem index — must not duplicate.
    save_defense_results(_ppl(), _ppl(2.0), str(tmp_path), problem_idx=0)
    save_defense_results(_ppl(), _ppl(3.0), str(tmp_path), problem_idx=0)
    path = tmp_path / "defense" / "defense_results.jsonl"
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 1
    assert rows[0]["harmful_after"]["perplexity_avg"] == [3.0]


def test_rows_sorted_by_problem_idx(tmp_path):
    # Append in reverse order; the file should still come out sorted.
    save_defense_results(_ppl(), _ppl(), str(tmp_path), problem_idx=2)
    save_defense_results(_ppl(), _ppl(), str(tmp_path), problem_idx=0)
    save_defense_results(_ppl(), _ppl(), str(tmp_path), problem_idx=1)
    path = tmp_path / "defense" / "defense_results.jsonl"
    rows = [json.loads(line) for line in open(path)]
    assert [r["problem_idx"] for r in rows] == [0, 1, 2]


def test_clean_holdout_optional(tmp_path):
    save_defense_results(_ppl(), _ppl(), str(tmp_path), problem_idx=0)
    rows = [
        json.loads(line)
        for line in open(tmp_path / "defense" / "defense_results.jsonl")
    ]
    assert "clean_before" not in rows[0]
    assert "clean_after" not in rows[0]


def test_schema_includes_required_keys(tmp_path):
    save_defense_results(
        _ppl(),
        _ppl(),
        str(tmp_path),
        problem_idx=0,
        clean_ppl_before=_ppl(),
        clean_ppl_after=_ppl(),
    )
    row = json.loads(
        (tmp_path / "defense" / "defense_results.jsonl").read_text().strip()
    )
    assert set(row) == {
        "problem_idx",
        "harmful_before",
        "harmful_after",
        "clean_before",
        "clean_after",
    }
