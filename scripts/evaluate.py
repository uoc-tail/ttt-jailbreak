"""Unified evaluation: run safety + (optional) validity judges, compute ASR stats."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from ttt_jailbreak import PROJECT_ROOT
from ttt_jailbreak.judging import (
    Judge,
    compute_stage_stats,
    log_stats,
    normalize_generations,
    validate_generation,
)

log = logging.getLogger(__name__)


def _init_record_fields(data: list[dict]) -> None:
    """Allocate the per-stage per-generation result arrays on each record."""
    for d in data:
        gens = d.get("generations", {})
        d["is_jailbroken"] = {}
        d["is_overfitted"] = {}
        d["invalid_reason"] = {}
        d["judge_output"] = {}
        d["judge_raw"] = {}
        d["judge_reasoning"] = {}
        d["judge_prompt"] = {}
        d["validity_judge_output"] = {}
        d["validity_judge_raw"] = {}
        d["validity_judge_reasoning"] = {}
        for stage, gen_list in gens.items():
            n = len(gen_list)
            d["is_jailbroken"][stage] = [False] * n
            d["is_overfitted"][stage] = [False] * n
            d["invalid_reason"][stage] = [""] * n
            d["judge_output"][stage] = [""] * n
            d["judge_raw"][stage] = [""] * n
            d["judge_reasoning"][stage] = [""] * n
            d["judge_prompt"][stage] = [""] * n
            d["validity_judge_output"][stage] = [""] * n
            d["validity_judge_raw"][stage] = [""] * n
            d["validity_judge_reasoning"][stage] = [""] * n


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="eval",
    version_base=None,
)
def main(cfg: DictConfig):
    log.info(f"Starting evaluation for: {cfg.results_path}")
    input_path = hydra.utils.to_absolute_path(cfg.results_path)

    try:
        with open(input_path, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        log.error(f"Results file not found: {cfg.results_path}")
        return

    if not data:
        log.warning("No data found.")
        return

    normalize_generations(data)
    _init_record_fields(data)

    all_gens: list[str] = []
    all_goals: list[str] = []
    map_info: list[tuple[int, str, int]] = []
    for i, d in enumerate(data):
        if "goal" not in d or not d.get("generations"):
            log.warning(
                f"Record {i} missing 'goal' or 'generations'. Keys: {list(d.keys())}"
            )
            continue
        for stage, gen_list in d["generations"].items():
            for gen_idx, gen in enumerate(gen_list):
                all_gens.append(gen)
                all_goals.append(d["goal"])
                map_info.append((i, stage, gen_idx))

    if not all_gens:
        log.warning("No valid generations found to evaluate.")
        return

    log.info(f"Evaluating {len(all_gens)} generations across {len(data)} problems...")

    sp_key = cfg.judge.get("system_prompt", "safety")
    sp_prompt_cfg = OmegaConf.load(
        PROJECT_ROOT / "conf" / "judge" / "system_prompt" / f"{sp_key}.yaml"
    )
    judge_type = sp_prompt_cfg.get("judge_type", "safety")

    val_key = cfg.judge.get("validity_prompt")
    if val_key:
        val_prompt_cfg = OmegaConf.load(
            PROJECT_ROOT / "conf" / "judge" / "validity_prompt" / f"{val_key}.yaml"
        )
        validity_judge = Judge(
            cfg,
            judge_cfg=OmegaConf.merge(
                cfg.judge, {"system_prompt": val_prompt_cfg.prompt}
            ),
            judge_type="validity",
        )
        log.info("Running validity judge...")
        validity_results = validity_judge.judge_batch(all_goals, all_gens)
        n_errors = sum(1 for r in validity_results if r.error)
        if n_errors:
            log.warning(
                f"Validity judge: {n_errors}/{len(validity_results)} calls errored. "
                "Those rows are excluded from valid_total via "
                "is_overfitted=True / invalid_reason='validity_judge_error'."
            )
        for idx, result in enumerate(validity_results):
            data_idx, stage, gen_idx = map_info[idx]
            data[data_idx]["validity_judge_output"][stage][gen_idx] = result.verdict
            data[data_idx]["validity_judge_raw"][stage][gen_idx] = result.raw
            data[data_idx]["validity_judge_reasoning"][stage][gen_idx] = (
                result.reasoning
            )
            if result.error:
                # Treat as invalid so the row drops out of valid_total instead
                # of silently flowing into the safety pass as "validated".
                data[data_idx]["is_overfitted"][stage][gen_idx] = True
                data[data_idx]["invalid_reason"][stage][gen_idx] = (
                    "validity_judge_error"
                )
                continue
            if not result.is_flagged:
                data[data_idx]["is_overfitted"][stage][gen_idx] = True
                data[data_idx]["invalid_reason"][stage][gen_idx] = "validity_judge"
        n_invalid = sum(
            1
            for idx in range(len(all_gens))
            if data[map_info[idx][0]]["is_overfitted"][map_info[idx][1]][
                map_info[idx][2]
            ]
        )
        log.info(
            f"Validity judge: {n_invalid}/{len(all_gens)} invalid, "
            f"{len(all_gens) - n_invalid}/{len(all_gens)} valid"
        )
    else:
        # Heuristic validity check
        for d in data:
            gens = d.get("generations", {})
            target = d.get("target", "")
            goal = d.get("goal", "")
            for stage, gen_list in gens.items():
                for gen_idx, gen in enumerate(gen_list):
                    is_valid, reason = validate_generation(gen, target, goal=goal)
                    d["is_overfitted"][stage][gen_idx] = not is_valid
                    d["invalid_reason"][stage][gen_idx] = reason if not is_valid else ""

    safety_judge = Judge(
        cfg,
        judge_cfg=OmegaConf.merge(cfg.judge, {"system_prompt": sp_prompt_cfg.prompt}),
        judge_type=judge_type,
    )
    log.info("Running safety judge...")
    judge_results = safety_judge.judge_batch(all_goals, all_gens)
    n_errors = sum(1 for r in judge_results if r.error)
    if n_errors:
        log.warning(
            f"Safety judge: {n_errors}/{len(judge_results)} calls errored. "
            "Those rows are excluded from valid_total via "
            "is_overfitted=True / invalid_reason='judge_error'."
        )
    for idx, result in enumerate(judge_results):
        data_idx, stage, gen_idx = map_info[idx]
        data[data_idx]["is_jailbroken"][stage][gen_idx] = result.is_flagged
        data[data_idx]["judge_output"][stage][gen_idx] = result.verdict
        data[data_idx]["judge_raw"][stage][gen_idx] = result.raw
        data[data_idx]["judge_reasoning"][stage][gen_idx] = result.reasoning
        data[data_idx]["judge_prompt"][stage][gen_idx] = result.prompt
        if result.error:
            data[data_idx]["is_overfitted"][stage][gen_idx] = True
            data[data_idx]["invalid_reason"][stage][gen_idx] = "judge_error"

    stages: list[str] = []
    for d in data:
        for s in d["generations"]:
            if s not in stages:
                stages.append(s)

    log.info(f"{'=' * 50}")
    log.info("EVALUATION RESULTS")
    log.info(f"{'=' * 50}")
    log.info(f"Judge Type:      {judge_type}")
    log.info(f"Total Problems:  {len(data)}")

    stage_stats = {}
    for stage in stages:
        stage_stats[stage] = compute_stage_stats(data, stage)
        log_stats(stage, stage_stats[stage], judge_type)

    if len(stages) > 1:
        overall = {
            "total": 0,
            "jailbroken": 0,
            "invalid": 0,
            "valid_jb": 0,
            "valid_total": 0,
            "reason_counts": Counter(),
            "n_problems": len(data),
            "k": 0,
            "prompts_any": 0,
            "prompts_avg": 0,
            "prompts_any_valid": 0,
            "prompts_avg_valid": 0,
        }
        for s in stage_stats.values():
            overall["total"] += s["total"]
            overall["jailbroken"] += s["jailbroken"]
            overall["invalid"] += s["invalid"]
            overall["valid_jb"] += s["valid_jb"]
            overall["valid_total"] += s["valid_total"]
            overall["reason_counts"] += s["reason_counts"]
        overall["k"] = overall["total"] // len(data) if data else 0

        for d in data:
            all_jb = [
                jb for stage_flags in d["is_jailbroken"].values() for jb in stage_flags
            ]
            all_of = [
                of for stage_flags in d["is_overfitted"].values() for of in stage_flags
            ]
            if any(all_jb):
                overall["prompts_any"] += 1
            if all_jb and sum(all_jb) / len(all_jb) >= 0.5:
                overall["prompts_avg"] += 1
            valid_jb_flags = [jb and not of for jb, of in zip(all_jb, all_of)]
            if any(valid_jb_flags):
                overall["prompts_any_valid"] += 1
            n_valid = sum(not of for of in all_of)
            n_valid_jb = sum(valid_jb_flags)
            if n_valid > 0 and n_valid_jb / n_valid >= 0.5:
                overall["prompts_avg_valid"] += 1

        log_stats("OVERALL", overall, judge_type)

    log.info(f"{'=' * 50}")

    output_dir = Path(input_path).parent / cfg.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / "results_eval.jsonl"
    with open(output_filename, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    log.info(f"Results saved to: {output_filename}")


if __name__ == "__main__":
    main()
