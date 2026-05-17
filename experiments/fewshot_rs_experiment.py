"""Few-shot TTT + Random-Search experiment.

For each problem, evaluates six conditions:

  1. no_attack_simple:         no TTT, no RS, plain prompt
  2. no_attack_refined:        no TTT, no RS, ``refined_best`` jailbreak prompt
  3. rs_refined:               no TTT, RS on the refined_best base-model prompt
  4. ttt_s{N}_refined:         TTT N steps, ``refined_best`` prompt at inference
  5. ttt_s{N}_base_rs_refined: TTT N steps, refined_best augmented with the
                               RS suffix found on the base model (condition 3)
  6. ttt_s{N}_rs_refined:      TTT N steps, RS on the TTT'd model, then generate

The backend's persistent LoRA adapter doubles as the base/TTT'd toggle:
``backend.reset()`` zeroes the adapter (base state), ``backend.fit()`` applies
TTT, and ``backend.model`` exposes the HF model for the RS gradient passes.

Paper hyper-parameters (see README for the canonical Llama3-8B command).
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
from ttt_jailbreak.attack import (
    get_target_str_for_goal,
    get_universal_manual_prompt,
    insert_suffix,
    run_random_search_on_backend,
)
from ttt_jailbreak.data import (
    example_from_problem,
    sample_few_shot_examples,
)
from ttt_jailbreak.utils import seed_everything

log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="random_search",
    version_base=None,
)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

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

    system_prompt = cfg.data.get("system_prompt", "")
    max_length = cfg.max_prompt_length + cfg.generate.max_new_tokens
    target_prefix = "<text>\n"  # for refined_best
    n_steps = cfg.attack.ttt_steps

    gen_extra = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.9,
    }
    num_samples = 10
    max_new_tokens = 512

    all_results: list[dict] = []
    for problem_idx in range(len(problems)):
        global_idx = cfg.start_idx + problem_idx
        seed_everything(cfg.seed + global_idx)

        problem = problems[problem_idx]
        goal = problem["goal"]
        target_str = problem.get("target", get_target_str_for_goal(goal))
        log.info(f"\n{'=' * 60}")
        log.info(f"Problem {global_idx}: {goal[:80]}...")

        refined_user_content = get_universal_manual_prompt(
            prompt_template="refined_best",
            target_str=target_str,
            goal=goal.lower(),
        )

        record: dict = {
            "problem_idx": global_idx,
            "goal": goal,
            "target": target_str,
            "user_contents": {
                "simple": goal,
                "refined_best": refined_user_content,
            },
            "generations": {},
        }

        # ── Base model: zero LoRA adapter → backend behaves as base ─────────
        backend.reset(seed=cfg.seed + global_idx)

        log.info("  [no_attack_simple]")
        record["generations"]["no_attack_simple"] = backend.generate(
            goal=goal,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_extra,
        )
        log.info("  [no_attack_refined]")
        record["generations"]["no_attack_refined"] = backend.generate(
            goal=refined_user_content,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_extra,
        )

        log.info("  [rs_refined] Running RS on base model...")
        base_rs_suffix, base_rs_loss = run_random_search_on_backend(
            backend,
            user_content=refined_user_content,
            target_str=target_str,
            target_prefix=target_prefix,
            system_prompt=system_prompt,
            cfg=cfg,
        )
        log.info(f"  RS suffix='{base_rs_suffix[:60]}...' loss={base_rs_loss:.4f}")
        base_rs_user_content = insert_suffix(
            refined_user_content,
            base_rs_suffix,
            position="end",
        )
        record["generations"]["rs_refined"] = backend.generate(
            goal=base_rs_user_content,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_extra,
        )
        record["user_contents"]["rs_refined"] = base_rs_user_content
        record["rs_suffix"] = base_rs_suffix
        record["rs_loss"] = base_rs_loss

        # ── TTT few-shot ────────────────────────────────────────────────────
        few_shot_indices = sample_few_shot_examples(
            full_dataset=full_dataset,
            target_idx=global_idx,
            num_shots=5,
            seed=cfg.seed + global_idx,
        )
        log.info(f"  [TTT s{n_steps}] few_shot_indices={few_shot_indices}")
        examples = [
            example_from_problem(dict(full_dataset[i])) for i in few_shot_indices
        ]
        losses = backend.fit(
            examples=examples,
            finetune_mode="few_shot",
            n_steps=cfg.trainer.args.max_steps,
            lr=cfg.trainer.args.learning_rate,
            max_length=max_length,
        )
        log.info(f"  TTT losses: {[f'{loss:.4f}' for loss in losses]}")
        record[f"ttt_s{n_steps}_info"] = {
            "losses": losses,
            "few_shot_indices": few_shot_indices,
            "few_shot_goals": [g for g, _ in examples],
            "hparams": {
                "lr": cfg.trainer.args.learning_rate,
                "r": cfg.trainer.peft_config.r,
                "lora_alpha": cfg.trainer.peft_config.lora_alpha,
                "n_steps": n_steps,
            },
        }

        log.info(f"  [ttt_s{n_steps}_refined]")
        record["generations"][f"ttt_s{n_steps}_refined"] = backend.generate(
            goal=refined_user_content,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_extra,
        )

        log.info(f"  [ttt_s{n_steps}_base_rs_refined]")
        record["generations"][f"ttt_s{n_steps}_base_rs_refined"] = backend.generate(
            goal=base_rs_user_content,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_extra,
        )
        record["user_contents"][f"ttt_s{n_steps}_base_rs_refined"] = (
            base_rs_user_content
        )

        log.info(f"  [ttt_s{n_steps}_rs_refined] Running RS on TTT'd model...")
        ttt_rs_suffix, ttt_rs_loss = run_random_search_on_backend(
            backend,
            user_content=refined_user_content,
            target_str=target_str,
            target_prefix=target_prefix,
            system_prompt=system_prompt,
            cfg=cfg,
        )
        log.info(f"  RS suffix='{ttt_rs_suffix[:60]}...' loss={ttt_rs_loss:.4f}")
        ttt_rs_user_content = insert_suffix(
            refined_user_content,
            ttt_rs_suffix,
            position="end",
        )
        record["generations"][f"ttt_s{n_steps}_rs_refined"] = backend.generate(
            goal=ttt_rs_user_content,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            **gen_extra,
        )
        record["user_contents"][f"ttt_s{n_steps}_rs_refined"] = ttt_rs_user_content
        record[f"ttt_s{n_steps}_rs_suffix"] = ttt_rs_suffix
        record[f"ttt_s{n_steps}_rs_loss"] = ttt_rs_loss

        all_results.append(record)
        with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
            for res in all_results:
                f.write(json.dumps(res) + "\n")

    log.info(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
