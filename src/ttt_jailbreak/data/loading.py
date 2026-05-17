"""Dataset loading + per-row formatting.

The single public function ``load_and_format_dataset`` is referenced as a
Hydra ``_target_`` in every ``conf/data/*.yaml``. It returns a HuggingFace
Dataset enriched with ``prompt`` (chat-templated), ``goal`` (raw input), and
``user_content`` (post-template-wrapping, pre-chat-template).
"""

from __future__ import annotations

from typing import List, Optional

from datasets import Dataset, load_dataset


def load_and_format_dataset(
    path: str,
    name: Optional[str] = None,
    split: str = "test",
    tokenizer=None,
    system_prompt: str = "You are a helpful assistant.",
    input_column: str = "question",
    fallback_column: Optional[str] = None,
    target_column: Optional[str] = None,
    keep_columns: Optional[List[str]] = None,
    column_mapping: Optional[dict] = None,
    data_files: Optional[str] = None,
    field: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_null_prompts: bool = False,
    enable_thinking: Optional[bool] = None,
    suffix: Optional[str] = None,
    universal_prompt_template: Optional[str] = None,
    suffix_position: str = "end",
    **kwargs,  # absorb extra Hydra keys
) -> Dataset:
    """Load a HF dataset and add ``prompt``/``goal``/``user_content`` columns.

    Each row gets:
      - ``prompt``: tokenizer.apply_chat_template applied to the (optional
        system + user) messages, ready to feed to ``model.generate``.
      - ``goal``: the raw ``input_column`` value.
      - ``user_content``: ``goal`` after optional ``universal_prompt_template``
        wrapping and optional ``suffix`` insertion; the literal user-turn
        content that was fed into the chat template.
    """
    load_kwargs = {}
    if data_files:
        load_kwargs["data_files"] = data_files
    if field:
        load_kwargs["field"] = field

    if data_files or not name:
        dataset = load_dataset(path, split=split, **load_kwargs)
    else:
        dataset = load_dataset(path, name, split=split, **load_kwargs)

    if filter_null_prompts and filter_column:
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: x.get(filter_column) is not None)
        if len(dataset) != original_len:
            print(f"Filtered {original_len - len(dataset)} null rows")

    if universal_prompt_template:
        from ttt_jailbreak.attack.prompts import (
            get_target_str_for_goal,
            get_universal_manual_prompt,
            insert_suffix,
        )

    def format_fn(example):
        raw_input = example.get(input_column)
        was_jailbroken = raw_input is not None
        if raw_input is None:
            raw_input = example.get(fallback_column, "") if fallback_column else ""
        original_raw_input = raw_input

        if universal_prompt_template:
            target_str = get_target_str_for_goal(raw_input)
            raw_input = get_universal_manual_prompt(
                universal_prompt_template,
                target_str,
                raw_input.lower(),
            )

        if suffix:
            if universal_prompt_template:
                raw_input = insert_suffix(raw_input, suffix, suffix_position)
            elif suffix_position == "end":
                raw_input = raw_input + " " + suffix
            else:
                raw_input = suffix + " " + raw_input

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": raw_input})

        chat_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if enable_thinking is not None:
            chat_kwargs["enable_thinking"] = enable_thinking

        result = {
            "prompt": tokenizer.apply_chat_template(messages, **chat_kwargs),
            "goal": original_raw_input,
            "user_content": raw_input,
            "was_jailbroken": was_jailbroken,
        }
        if target_column:
            if target_column not in example:
                raise KeyError(
                    f"target_column={target_column!r} not found in example. "
                    f"Available columns: {list(example.keys())}. "
                    "Pass target_column=None to skip target loading, or fix "
                    "the dataset config to reference the right column."
                )
            result["target"] = example[target_column]

        if keep_columns:
            mapping = column_mapping or {}
            for col in keep_columns:
                if col in example and col not in result:
                    result[mapping.get(col, col)] = example[col]
        return result

    return dataset.map(format_fn)
