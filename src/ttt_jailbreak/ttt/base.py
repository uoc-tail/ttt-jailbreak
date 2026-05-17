"""Test-time training (TTT) backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

FinetuneMode = Literal["prompt", "target", "few_shot"]


class TTTBackend(ABC):
    """Backend that runs per-problem test-time training.

    Lifecycle: construct once → (reset → fit → generate) per problem. The
    constructor fixes everything that does not change across problems
    (model identity, system prompt, max length, peft config). The per-problem
    methods take only what varies: (goal, target) pairs and training
    hyper-parameters.

    Subclasses must override ``fit``, ``generate``, and ``reset``. ``perplexity``
    is optional and gated by the ``supports_perplexity`` class attribute, which
    the defense module checks before calling.
    """

    #: Whether ``perplexity()`` is implemented. The defense module hard-errors
    #: against backends that do not advertise support.
    supports_perplexity: bool = False

    @abstractmethod
    def fit(
        self,
        examples: list[tuple[str, str]],
        finetune_mode: FinetuneMode,
        *,
        n_steps: int,
        lr: float,
        max_length: int | None = None,
    ) -> list[float]:
        """Fit on (goal, target) pairs. Returns per-step training losses.

        ``examples`` has length 1 for single-problem TTT and length N for
        few-shot TTT. ``finetune_mode`` selects which tokens contribute to the
        loss (see config under ``conf/finetune_mode/``).
        """

    @abstractmethod
    def generate(
        self,
        goal: str | None = None,
        *,
        messages: list[dict] | None = None,
        num_samples: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> list[str]:
        """Sample ``num_samples`` completions.

        Pass either ``goal`` (a raw user turn; the backend prepends its
        configured system prompt) or ``messages`` (an explicit chat history,
        e.g., for in-context-learning baselines).
        """

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Discard any fit() effects, restoring the model to its base state.

        ``seed`` (optional) is applied before any stochastic reset (e.g., LoRA
        adapter re-initialization) so per-problem state is reproducible.
        """

    def apply_chat_template(self, messages: list[dict]) -> str:
        """Render ``messages`` as the chat-templated prompt string.

        Used for diagnostics (e.g., recording the prompt fed to ``generate``
        in result files) and for callers that need to build multi-turn
        prompts before invoking ``generate(messages=...)``.

        Default implementation calls ``self.tokenizer.apply_chat_template``;
        backends without a HuggingFace-compatible tokenizer must override.
        """
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        enable_thinking = getattr(self, "enable_thinking", None)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def perplexity(self, *args, **kwargs):
        """Optional: compute per-example perplexity. Override in subclasses
        that set ``supports_perplexity = True``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement perplexity()."
        )
