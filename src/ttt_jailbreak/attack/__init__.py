"""Jailbreak attack primitives: prompt templates + random search."""

from .prompts import (
    LLAMA2_DEFAULT_SYSTEM_PROMPT,
    apply_chat_template,
    get_target_str_for_goal,
    get_universal_manual_prompt,
    insert_suffix,
    strip_chat_template_suffix,
)
from .random_search import (
    get_template_parts,
    run_random_search,
    run_random_search_on_backend,
)

__all__ = [
    "LLAMA2_DEFAULT_SYSTEM_PROMPT",
    "apply_chat_template",
    "get_target_str_for_goal",
    "get_template_parts",
    "get_universal_manual_prompt",
    "insert_suffix",
    "run_random_search",
    "run_random_search_on_backend",
    "strip_chat_template_suffix",
]
