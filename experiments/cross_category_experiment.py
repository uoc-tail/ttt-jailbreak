"""Cross-category TTT transfer experiment.

For a given ``source_category``, sample 5 few-shot examples from that macro
category, TTT the model on them, then generate for ALL problems. The
(source × target) heatmap reveals whether the TTT effect transfers across
categories or is category-specific.

For defensive research purposes only.
"""

from __future__ import annotations

import json
import logging
import os
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ttt_jailbreak import PROJECT_ROOT
from ttt_jailbreak.data import (
    example_from_problem,
    get_macro_category,
)
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
            "cross_category requires finetune_mode=few_shot "
            "(it samples 5 shots per source category)."
        )

    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    log.info("Instantiating TTT backend...")
    backend = instantiate(cfg.backend, _recursive_=True)

    log.info(f"Loading dataset {cfg.data.name}...")
    full_dataset = instantiate(cfg.data, tokenizer=backend.tokenizer)
    end_idx = (
        cfg.start_idx + cfg.num_problems if cfg.num_problems else len(full_dataset)
    )
    problems = full_dataset.select(range(cfg.start_idx, end_idx))
    log.info(f"Processing {len(problems)} problems")

    num_shots = cfg.finetune_mode.get("num_shots", 5)
    max_length = cfg.max_prompt_length + cfg.generate.max_new_tokens

    macro_to_indices: dict[str, list[int]] = {}
    for i in range(len(problems)):
        raw_cat = problems[i].get("category", "unknown")
        macro = get_macro_category(raw_cat)
        macro_to_indices.setdefault(macro, []).append(cfg.start_idx + i)

    macro_categories = sorted(macro_to_indices.keys())
    log.info(
        "Macro categories: "
        + ", ".join(f"{m} ({len(macro_to_indices[m])})" for m in macro_categories)
    )

    source_examples: dict[str, list[int]] = {}
    for macro in macro_categories:
        rng = random.Random(cfg.seed)
        indices = macro_to_indices[macro]
        k = min(num_shots, len(indices))
        source_examples[macro] = rng.sample(indices, k)
        log.info(f"Source '{macro}': selected {k} examples: {source_examples[macro]}")

    source_category = cfg.source_category
    assert source_category in macro_to_indices, (
        f"Unknown source_category '{source_category}'. "
        f"Available: {list(macro_to_indices.keys())}"
    )
    fs_indices = source_examples[source_category]
    log.info(f"Source category: {source_category} (examples: {fs_indices})")

    examples = [example_from_problem(dict(full_dataset[i])) for i in fs_indices]

    log.info(f"TTT training on {len(examples)} source examples...")
    backend.reset(seed=cfg.seed)
    losses = backend.fit(
        examples=examples,
        finetune_mode="few_shot",
        n_steps=cfg.trainer.args.max_steps,
        lr=cfg.trainer.args.learning_rate,
        max_length=max_length,
    )
    log.info(f"TTT losses: {[f'{loss:.4f}' for loss in losses]}")

    gen_kwargs = OmegaConf.to_container(cfg.generate, resolve=True)
    max_new_tokens = gen_kwargs.pop("max_new_tokens")
    num_samples = gen_kwargs.pop("num_return_sequences", 10)

    all_results: list[dict] = []
    for problem_idx in range(len(problems)):
        global_idx = cfg.start_idx + problem_idx
        seed_everything(cfg.seed + global_idx)
        problem = problems[problem_idx]
        goal_for_gen, _ = example_from_problem(problem)
        prompt_text = problem[cfg.data.get("prompt_key", "prompt")]
        raw_cat = problem.get("category", "unknown")
        target_macro = get_macro_category(raw_cat)

        log.info(
            f"Problem {global_idx} [{target_macro}]: {problem.get('goal', '')[:60]}..."
        )
        generations = backend.generate(
            goal=goal_for_gen,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

        result_record = {
            "problem_idx": global_idx,
            "source_category": source_category,
            "source_examples": fs_indices,
            "target_category": target_macro,
            "goal": problem.get("goal", ""),
            "target": problem.get("target", ""),
            "category": raw_cat,
            "prompt": prompt_text,
            "generations": generations,
            "ttt_losses": losses,
        }
        all_results.append(result_record)
        with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
            for res in all_results:
                f.write(json.dumps(res) + "\n")

    log.info(f"Done. Results saved to {output_dir}")
    return cfg.timestamp


if __name__ == "__main__":
    main()
