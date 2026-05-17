"""In-context-learning (ICL) few-shot baseline.

For each problem, builds a multi-turn chat with k few-shot (goal, target) pairs
as prior user/assistant turns, then generates from the actual goal. No
training: the few-shot examples enter only as context.

This script reuses the TTTBackend solely as a model + tokenizer loader; fit/reset
are never called. If ICL alone breaks alignment, TTT is unnecessary; if not,
the adversarial weight update is what defeats safety training.

For defensive research purposes only.
"""

from __future__ import annotations

import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ttt_jailbreak import PROJECT_ROOT
from ttt_jailbreak.data import sample_few_shot_examples
from ttt_jailbreak.utils import seed_everything

log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="jailbreak",
    version_base=None,
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    if cfg.get("finetune_mode") is None or cfg.finetune_mode.get("name") != "few_shot":
        raise ValueError(
            "icl_fewshot requires finetune_mode=few_shot to read num_shots."
        )

    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    log.info("Instantiating TTT backend (model+tokenizer only; fit unused)...")
    backend = instantiate(cfg.backend, _recursive_=True)

    log.info(f"Loading dataset {cfg.data.name}...")
    full_dataset = instantiate(cfg.data, tokenizer=backend.tokenizer)
    end_idx = (
        cfg.start_idx + cfg.num_problems if cfg.num_problems else len(full_dataset)
    )
    problems = full_dataset.select(range(cfg.start_idx, end_idx))
    log.info(
        f"Processing {len(problems)} problems (indices {cfg.start_idx} to {end_idx - 1})"
    )

    system_prompt = cfg.data.get("system_prompt", "")
    num_shots = cfg.finetune_mode.get("num_shots", 5)

    gen_kwargs = OmegaConf.to_container(cfg.generate, resolve=True)
    max_new_tokens = gen_kwargs.pop("max_new_tokens")
    num_samples = gen_kwargs.pop("num_return_sequences", 10)

    all_results: list[dict] = []
    for problem_idx in range(len(problems)):
        global_idx = cfg.start_idx + problem_idx
        seed_everything(cfg.seed + global_idx)
        problem = problems[problem_idx]
        goal = problem.get("goal", "")
        target = problem.get("target", "")

        log.info(f"{'=' * 20} Problem {global_idx} {'=' * 20}")

        few_shot_indices = sample_few_shot_examples(
            full_dataset=full_dataset,
            target_idx=global_idx,
            num_shots=num_shots,
            seed=cfg.seed + global_idx,
        )
        log.info(f"Few-shot indices: {few_shot_indices}")

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        few_shot_goals: list[str] = []
        few_shot_targets: list[str] = []
        for idx in few_shot_indices:
            fs_goal = full_dataset[idx].get("goal", "")
            fs_target = full_dataset[idx].get("target", "")
            few_shot_goals.append(fs_goal)
            few_shot_targets.append(fs_target)
            messages.append({"role": "user", "content": fs_goal})
            messages.append({"role": "assistant", "content": fs_target})

        messages.append({"role": "user", "content": goal})

        log.info(f"Generating with ICL prompt ({len(messages)} messages)...")
        generations = backend.generate(
            messages=messages,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

        # Use the backend's chat template to record the exact prompt used.
        icl_prompt = backend.apply_chat_template(messages)

        result_record = {
            "problem_idx": global_idx,
            "prompt": icl_prompt,
            "generations": generations,
            "goal": goal,
            "target": target,
            "few_shot_info": {
                "num_shots": num_shots,
                "few_shot_indices": few_shot_indices,
                "few_shot_goals": few_shot_goals,
                "few_shot_targets": few_shot_targets,
            },
        }

        column_mapping = cfg.data.get("column_mapping", {})
        for key in cfg.data.get("metadata_keys", []):
            val = problem.get(key, problem.get(key.lower(), None))
            final_key = column_mapping.get(key, key)
            if final_key not in result_record:
                result_record[final_key] = val

        all_results.append(result_record)
        with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
            for res in all_results:
                f.write(json.dumps(res) + "\n")

    log.info(f"Job complete. Results saved to {output_dir}")
    return cfg.timestamp


if __name__ == "__main__":
    main()
