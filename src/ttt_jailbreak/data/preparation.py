"""Per-example TTT data preparation (tokenization + loss-mask construction)."""

from __future__ import annotations

import re
from typing import List, Literal, Optional, Set

from datasets import Dataset

LossMode = Literal["full", "target", "prompt", "few_shot"]


# Tokens that control conversation flow (role markers, turn delimiters, thinking
# tags). In a realistic TTT API these are masked from the loss so fine-tuning
# cannot teach the model to emit/skip structural tokens.
_STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<s>",
    "</s>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<think>",
    "</think>",
)


def get_special_token_ids(tokenizer) -> Set[int]:
    """Return token IDs that should be masked from the TTT loss."""
    special_ids: Set[int] = set()

    for attr in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(token_id)

    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)

    for s in _STRUCTURAL_TOKENS:
        try:
            encoded = tokenizer.encode(s, add_special_tokens=False)
            if len(encoded) == 1:
                special_ids.add(encoded[0])
        except Exception:
            pass

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for token, token_id in vocab.items():
        if token.startswith("<|") and token.endswith("|>"):
            special_ids.add(token_id)
        elif re.match(r"^(system|user|assistant)$", token, re.IGNORECASE):
            special_ids.add(token_id)

    return special_ids


def prepare_ttt_data(
    problem: dict,
    prompt_template: str,
    completion_template: str = "",
    name: LossMode = "target",
    tokenizer=None,
    max_length: Optional[int] = None,
    mask_special_tokens: bool = True,
    strip_prompt_suffix: bool = False,
    append_eos: bool = False,
    answer_only: bool = False,
) -> Dataset:
    """Build a one-row HF Dataset with ``input_ids``, ``attention_mask``, ``labels``.

    ``name`` selects which positions contribute to the loss:
      - ``"target"``: completion (assistant turn) only
      - ``"prompt"``: user content only
      - ``"few_shot"``: user content + everything after
      - ``"full"``: every position (ablation)
    """
    if not isinstance(problem, dict):
        problem = dict(problem)

    prompt_text = problem.get("prompt", problem.get("Goal", ""))
    target = problem.get("target", problem.get("Target", ""))

    prompt_str = prompt_template.format(prompt=prompt_text, target=target)

    if strip_prompt_suffix and tokenizer is not None:
        from ttt_jailbreak.attack.prompts import strip_chat_template_suffix

        prompt_str = strip_chat_template_suffix(tokenizer, prompt_str)

    completion_str = (
        completion_template.format(prompt=prompt_text, target=target)
        if completion_template
        else ""
    )
    full_text = prompt_str + completion_str

    encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    special_ids = get_special_token_ids(tokenizer) if mask_special_tokens else set()

    if name == "full":
        labels = list(input_ids)

    elif name == "target":
        # Use the joint offset_mapping to find the prompt/target boundary.
        # Re-tokenizing prompt_str alone and slicing input_ids by its length is
        # wrong for BPE / SentencePiece tokenizers: the last prompt token may
        # merge with the leading completion bytes, so the isolated prompt
        # length does not equal the prompt prefix in the joint tokenization.
        boundary = len(prompt_str)
        labels = [
            tid if tok_start >= boundary else -100
            for tid, (tok_start, _) in zip(input_ids, offset_mapping)
        ]

    elif name == "prompt":
        user_content = (
            problem.get("user_content")
            or problem.get("goal")
            or problem.get("Goal")
            or ""
        )
        labels = _compute_content_labels(
            input_ids,
            offset_mapping,
            full_text,
            user_content,
            user_content,
        )

    elif name == "few_shot":
        user_content = (
            problem.get("user_content")
            or problem.get("goal")
            or problem.get("Goal")
            or ""
        )
        labels = _compute_content_labels(
            input_ids,
            offset_mapping,
            full_text,
            user_content,
            None,
        )

    else:
        raise ValueError(f"Unknown name/loss_mode: {name}")

    if mask_special_tokens:
        for i, token_id in enumerate(input_ids):
            if token_id in special_ids:
                labels[i] = -100

    if answer_only:
        prompt_only_len = len(
            tokenizer(
                problem.get("prompt", ""),
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
        )
        labels = [-100] * min(prompt_only_len, len(labels)) + labels[prompt_only_len:]

    # EOS appended last so mask_special_tokens does not strip it back out.
    if append_eos and tokenizer.eos_token_id is not None:
        eos_id = tokenizer.eos_token_id
        input_ids = input_ids + [eos_id]
        labels = labels + [eos_id]

    return Dataset.from_list(
        [
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        ]
    )


def _compute_content_labels(
    input_ids: List[int],
    offset_mapping: List[tuple],
    full_text: str,
    content_start: str,
    content_end: Optional[str],
) -> List[int]:
    start_idx = full_text.find(content_start)
    if start_idx == -1:
        return [-100] * len(input_ids)

    end_idx = start_idx + len(content_end) if content_end else len(full_text)

    labels = [-100] * len(input_ids)
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if content_end:
            if tok_start < end_idx and tok_end > start_idx:
                labels[i] = input_ids[i]
        else:
            if tok_start >= start_idx:
                labels[i] = input_ids[i]

    return labels
