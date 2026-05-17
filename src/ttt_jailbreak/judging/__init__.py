"""LLM-as-judge evaluation: prompt construction, server I/O, result aggregation."""

from .judge import Judge, JudgeResult, parse_verdict, validate_generation
from .results import compute_stage_stats, log_stats, normalize_generations
from .server import (
    build_judge_prompt,
    check_server_health,
    run_batch_chat_completions,
    run_chat_completion,
    run_local_inference,
)

__all__ = [
    "Judge",
    "JudgeResult",
    "build_judge_prompt",
    "check_server_health",
    "compute_stage_stats",
    "log_stats",
    "normalize_generations",
    "parse_verdict",
    "run_batch_chat_completions",
    "run_chat_completion",
    "run_local_inference",
    "validate_generation",
]
