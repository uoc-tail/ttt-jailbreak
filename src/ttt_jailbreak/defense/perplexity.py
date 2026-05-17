"""Per-example perplexity utilities for the defense module.

Compares perplexity of target completions before vs. after TTT on private
holdouts. An attack that lowers harmful perplexity while leaving clean
perplexity stable is the detection signal.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset

log = logging.getLogger(__name__)


def _per_batch_perplexity(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.Tensor,
    prefix_lengths: Optional[List[int]] = None,
) -> Dict[str, List[float]]:
    """Run one forward pass and return per-example PPL + NLL metrics."""
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = target_mask[:, 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_logits.size(0), -1)

    masked_loss = loss_per_token * shift_mask.float()
    target_counts = shift_mask.float().sum(dim=1)
    has_target = target_counts > 0  # rows with at least one target token
    avg_nll = masked_loss.sum(dim=1) / target_counts.clamp(min=1)
    avg_nll = torch.where(has_target, avg_nll, torch.full_like(avg_nll, float("nan")))

    # argmax on an all-zero mask returns 0; mask the row to NaN instead so
    # truncated / empty rows do not silently report position-0 loss as the
    # first-target NLL.
    first_target_idx = shift_mask.float().argmax(dim=1)
    first_nll = loss_per_token.gather(1, first_target_idx.unsqueeze(1)).squeeze(1)
    first_nll = torch.where(
        has_target, first_nll, torch.full_like(first_nll, float("nan"))
    )

    results = {
        "perplexity_avg": torch.exp(avg_nll).cpu().tolist(),
        "perplexity_first": torch.exp(first_nll).cpu().tolist(),
        "nll_avg": avg_nll.cpu().tolist(),
        "nll_first": first_nll.cpu().tolist(),
    }

    if prefix_lengths:
        shift_mask_float = shift_mask.float()
        target_cumsum = shift_mask_float.cumsum(dim=1)
        for N in prefix_lengths:
            prefix_mask = shift_mask_float * (target_cumsum <= N).float()
            prefix_has_target = prefix_mask.sum(dim=1) > 0
            prefix_counts = prefix_mask.sum(dim=1).clamp(min=1)
            prefix_nll = (loss_per_token * prefix_mask).sum(dim=1) / prefix_counts
            prefix_nll = torch.where(
                prefix_has_target,
                prefix_nll,
                torch.full_like(prefix_nll, float("nan")),
            )
            results[f"perplexity_prefix_{N}"] = torch.exp(prefix_nll).cpu().tolist()
            results[f"nll_prefix_{N}"] = prefix_nll.cpu().tolist()

    return results


def _tokenize_with_target_mask(
    tokenizer,
    prompt: str,
    target: str,
    max_length: Optional[int] = None,
) -> Dict[str, list]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    all_ids = prompt_ids + target_ids
    if max_length and len(all_ids) > max_length:
        all_ids = all_ids[:max_length]
    prompt_len = len(prompt_ids)
    target_mask = [False] * min(prompt_len, len(all_ids)) + [True] * max(
        0, len(all_ids) - prompt_len
    )
    return {
        "input_ids": all_ids,
        "attention_mask": [1] * len(all_ids),
        "target_mask": target_mask,
    }


def _collate_and_pad(
    batch: List[Dict[str, list]],
    pad_token_id: int,
    device,
) -> Dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    padded = {"input_ids": [], "attention_mask": [], "target_mask": []}
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        padded["input_ids"].append(ex["input_ids"] + [pad_token_id] * pad_len)
        padded["attention_mask"].append(ex["attention_mask"] + [0] * pad_len)
        padded["target_mask"].append(ex["target_mask"] + [False] * pad_len)
    return {
        "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            padded["attention_mask"], dtype=torch.long, device=device
        ),
        "target_mask": torch.tensor(
            padded["target_mask"], dtype=torch.bool, device=device
        ),
    }


def generate_clean_targets(
    model,
    tokenizer,
    dataset: Dataset,
    goal_column: str = "question",
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 25,
    min_new_tokens: int = 5,
    temperature: float = 1.0,
    batch_size: int = 8,
    enable_thinking: Optional[bool] = None,
) -> List[str]:
    """Sample reference target completions from the base model.

    These targets serve as the *clean* holdout reference: post-TTT perplexity
    on them should not change much if the TTT update is narrow.
    """
    model.eval()
    device = next(model.parameters()).device

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking

    prompts = []
    for example in dataset:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": example[goal_column]})
        prompts.append(tokenizer.apply_chat_template(messages, **chat_kwargs))

    all_targets: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=pad_token_id,
            )
        input_len = inputs.input_ids.shape[1]
        generated = tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
        )
        all_targets.extend(t.strip() for t in generated)
        log.info(
            f"Generated targets for {min(i + batch_size, len(prompts))}"
            f"/{len(prompts)} examples"
        )
    return all_targets


def compute_dataset_perplexity(
    model,
    tokenizer,
    dataset: Dataset,
    batch_size: int = 8,
    system_prompt: str = "You are a helpful assistant.",
    max_length: Optional[int] = None,
    prompt_column: Optional[str] = None,
    goal_column: str = "goal",
    target_column: str = "target",
    target_subfield: Optional[str] = None,
    targets: Optional[List[str]] = None,
    prefix_lengths: Optional[List[int]] = None,
    enable_thinking: Optional[bool] = None,
) -> Dict[str, List[float]]:
    """Per-example perplexity of targets conditioned on questions.

    Inputs can come from either a pre-formatted ``prompt_column``, or
    ``goal_column`` + chat-template, or an explicit ``targets`` list overriding
    ``target_column``.

    Returns a dict mapping metric name (``perplexity_avg``, ``perplexity_first``,
    ``nll_avg``, ``nll_first``, plus optional ``perplexity_prefix_{N}`` /
    ``nll_prefix_{N}``) to a list of per-example floats.
    """
    model.eval()
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    tokenized_examples = []
    for idx, example in enumerate(dataset):
        if targets is not None:
            target = targets[idx]
        else:
            target = example[target_column]
            if target_subfield and isinstance(target, dict):
                target = target[target_subfield]

        if prompt_column and prompt_column in example:
            prompt = example[prompt_column]
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": example[goal_column]})
            chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                chat_kwargs["enable_thinking"] = enable_thinking
            prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)

        tokenized_examples.append(
            _tokenize_with_target_mask(tokenizer, prompt, target, max_length)
        )

    all_results: Dict[str, List[float]] = {}
    for i in range(0, len(tokenized_examples), batch_size):
        batch = tokenized_examples[i : i + batch_size]
        padded = _collate_and_pad(batch, pad_token_id, device)
        batch_results = _per_batch_perplexity(
            model,
            padded["input_ids"],
            padded["attention_mask"],
            padded["target_mask"],
            prefix_lengths=prefix_lengths,
        )
        for key, vals in batch_results.items():
            all_results.setdefault(key, []).extend(vals)
        log.info(
            f"Processed {min(i + batch_size, len(tokenized_examples))}"
            f"/{len(tokenized_examples)} examples"
        )
    return all_results


def save_defense_results(
    ppl_before: Dict[str, List[float]],
    ppl_after: Dict[str, List[float]],
    output_dir: str,
    problem_idx: int,
    clean_ppl_before: Optional[Dict[str, List[float]]] = None,
    clean_ppl_after: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Write per-problem before/after perplexity to ``<output_dir>/defense/defense_results.jsonl``.

    If a row already exists for ``problem_idx`` it is overwritten in place, so
    re-running an experiment in the same output directory does not duplicate
    rows.
    """
    defense_dir = os.path.join(output_dir, "defense")
    os.makedirs(defense_dir, exist_ok=True)
    path = os.path.join(defense_dir, "defense_results.jsonl")

    record = {
        "problem_idx": problem_idx,
        "harmful_before": ppl_before,
        "harmful_after": ppl_after,
    }
    if clean_ppl_before is not None:
        record["clean_before"] = clean_ppl_before
    if clean_ppl_after is not None:
        record["clean_after"] = clean_ppl_after

    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("problem_idx") != problem_idx:
                    rows.append(row)
    rows.append(record)
    rows.sort(key=lambda r: r.get("problem_idx", 0))
    with open(path, "w") as f:
        for row in rows:
            # Coerce NaN/Inf to null so the JSONL is consumable by strict
            # parsers (jq, browsers, JSON libraries that reject non-standard
            # literals). _per_batch_perplexity emits NaN for empty target_mask
            # rows.
            f.write(_dumps_strict(row) + "\n")


def _dumps_strict(obj) -> str:
    """``json.dumps`` with NaN/Inf coerced to ``null``."""
    def replace(o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, list):
            return [replace(x) for x in o]
        if isinstance(o, dict):
            return {k: replace(v) for k, v in o.items()}
        return o

    return json.dumps(replace(obj), allow_nan=False)
