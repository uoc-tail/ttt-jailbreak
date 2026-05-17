"""TTT jailbreak attack: per-problem train-then-generate via a TTTBackend.

Backend (HuggingFace or Tinker) is selected via Hydra (``backend=hf|tinker``).
Defense (perplexity monitoring on private holdouts) is opt-in via
``defense=default defense.enabled=true`` and is HF-only. Tinker backends hit a
clear ``RuntimeError`` inside ``DefenseMonitor``.
"""

from __future__ import annotations

import json
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ttt_jailbreak import PROJECT_ROOT
from ttt_jailbreak.data import example_from_problem, sample_few_shot_examples
from ttt_jailbreak.defense import DefenseMonitor
from ttt_jailbreak.utils import seed_everything

log = logging.getLogger(__name__)

SUPPORTED_FINETUNE_MODES = {"prompt", "target", "few_shot"}


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="jailbreak",
    version_base=None,
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    finetune_mode = None
    num_shots = 0
    deterministic_sampling = True
    if cfg.get("finetune_mode"):
        finetune_cfg = OmegaConf.to_container(cfg.finetune_mode, resolve=True)
        finetune_cfg.pop("_target_", None)
        finetune_mode = finetune_cfg.get("name")
        num_shots = finetune_cfg.pop("num_shots", 0)
        deterministic_sampling = finetune_cfg.pop("deterministic_sampling", True)

    if cfg.get("trainer") is not None and finetune_mode is None:
        raise ValueError(
            "trainer is set but finetune_mode is missing. Either pass "
            "finetune_mode=<prompt|target|few_shot> or drop the trainer "
            "(use ~finetune_mode for a vanilla baseline)."
        )

    if finetune_mode is not None and finetune_mode not in SUPPORTED_FINETUNE_MODES:
        raise ValueError(
            f"finetune_mode={finetune_mode!r} is not supported. "
            f"Choose one of {sorted(SUPPORTED_FINETUNE_MODES)}."
        )

    # Defense + Tinker is unsupported. Detect by backend _target_ rather than
    # by instantiating the backend, so we fail before any Tinker ServiceClient
    # is constructed (and before any HF model is loaded).
    defense_cfg = cfg.get("defense")
    if defense_cfg and defense_cfg.get("enabled", False):
        backend_target = cfg.backend.get("_target_", "")
        if "TinkerBackend" in backend_target:
            raise RuntimeError(
                f"Defense (perplexity monitoring) requested but backend "
                f"{backend_target} does not support perplexity()."
            )

    output_dir = hydra.utils.to_absolute_path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output directory set to: {output_dir}")

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
    log.info(
        f"Processing {len(problems)} problems (indices {cfg.start_idx} to {end_idx - 1})"
    )

    defense = None
    if defense_cfg and defense_cfg.get("enabled", False):
        defense = DefenseMonitor(backend, defense_cfg, cfg.data)
        defense.baseline()
        log.info("Defense: baseline perplexity computed.")

    max_length = cfg.max_prompt_length + cfg.generate.max_new_tokens

    all_results: list[dict] = []
    for problem_idx in range(len(problems)):
        global_idx = cfg.start_idx + problem_idx
        seed_everything(cfg.seed + global_idx)
        log.info(f"{'=' * 20} Problem {global_idx} {'=' * 20}")
        problem = problems[problem_idx]
        goal_for_gen, target = example_from_problem(problem)
        prompt_text = problem[cfg.data.get("prompt_key", "prompt")]

        losses = None
        few_shot_info = None

        if cfg.get("trainer") is not None:
            backend.reset(seed=cfg.seed + global_idx)

            if num_shots > 0:
                few_shot_seed = (
                    cfg.seed + global_idx if deterministic_sampling else None
                )
                few_shot_indices = sample_few_shot_examples(
                    full_dataset=full_dataset,
                    target_idx=global_idx,
                    num_shots=num_shots,
                    seed=few_shot_seed,
                )
                log.info(
                    f"Few-shot mode: {num_shots} shots, indices={few_shot_indices}"
                )
                examples = [
                    example_from_problem(dict(full_dataset[i]))
                    for i in few_shot_indices
                ]
                few_shot_info = {
                    "num_shots": len(few_shot_indices),
                    "few_shot_indices": few_shot_indices,
                    "few_shot_goals": [g for g, _ in examples],
                    "few_shot_targets": [t for _, t in examples],
                }
            else:
                examples = [(goal_for_gen, target)]

            losses = backend.fit(
                examples=examples,
                finetune_mode=finetune_mode,
                n_steps=cfg.trainer.args.max_steps,
                lr=cfg.trainer.args.learning_rate,
                max_length=max_length,
            )
            log.info(f"TTT losses: {[f'{loss:.4f}' for loss in losses]}")

            if defense is not None:
                defense.after_fit(global_idx, output_dir)

        log.info(f"Generating for problem {global_idx}...")
        gen_kwargs = OmegaConf.to_container(cfg.generate, resolve=True)
        max_new_tokens = gen_kwargs.pop("max_new_tokens")
        num_samples = gen_kwargs.pop("num_return_sequences", 10)

        generations = backend.generate(
            goal=goal_for_gen,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

        result_record = {
            "problem_idx": global_idx,
            "prompt": prompt_text,
            "generations": generations,
            "finetune_mode": finetune_mode,
            "ttt_losses": losses,
        }
        if few_shot_info:
            result_record["few_shot_info"] = few_shot_info

        column_mapping = cfg.data.get("column_mapping", {})
        for key in cfg.data.get("metadata_keys", []):
            val = problem.get(key, problem.get(key.lower(), None))
            final_key = column_mapping.get(key, key)
            result_record[final_key] = val

        all_results.append(result_record)
        with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
            for res in all_results:
                f.write(json.dumps(res) + "\n")

    log.info(f"Job complete. Results saved to {output_dir}")
    return cfg.timestamp


if __name__ == "__main__":
    main()
