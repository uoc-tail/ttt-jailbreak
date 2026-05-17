"""HuggingFace transformers backend for TTT."""

from __future__ import annotations

import gc
import logging
import math
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.preparation import prepare_ttt_data
from ..defense.perplexity import compute_dataset_perplexity
from ..utils import seed_everything
from .base import FinetuneMode, TTTBackend
from .trainer import ttt_train

log = logging.getLogger(__name__)


def _reset_lora(peft_model) -> None:
    """Reset LoRA adapter to PEFT's default init (Kaiming-uniform A, zero B)."""
    for _, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for key in module.lora_A:
                torch.nn.init.kaiming_uniform_(
                    module.lora_A[key].weight, a=math.sqrt(5)
                )
            for key in module.lora_B:
                torch.nn.init.zeros_(module.lora_B[key].weight)


_FINETUNE_DEFAULTS: dict[str, dict[str, Any]] = {
    "prompt": {
        "prompt_template": "{prompt}",
        "completion_template": "",
    },
    "target": {
        "prompt_template": "{prompt}",
        "completion_template": "{target}",
    },
    "few_shot": {
        "prompt_template": "{prompt}{target}",
        "completion_template": "",
        "append_eos": True,
    },
}


class HuggingFaceBackend(TTTBackend):
    """HuggingFace transformers backend.

    LoRA mode (``peft_config`` provided): the base model is loaded once at
    construction and reused across problems; ``reset()`` reinitializes the
    adapter weights in-place. This avoids reloading multi-GB weights per
    problem.

    Full fine-tuning mode (``peft_config=None``): the model is reloaded from
    disk on every ``reset()``. Slower but necessary because there is no cheap
    way to roll back full-weight updates.
    """

    supports_perplexity = True

    def __init__(
        self,
        model_name: str,
        *,
        system_prompt: str = "",
        max_length: int | None = None,
        chat_template: str | None = None,
        enable_thinking: bool | None = None,
        peft_config: Any = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        train_args: dict | None = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.peft_config = peft_config
        self._model_kwargs = dict(model_kwargs or {})
        self._train_args = dict(train_args or {})

        tok_kwargs = {"trust_remote_code": True, "padding_side": "left"}
        tok_kwargs.update(tokenizer_kwargs or {})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self.tokenizer.add_eos_token = False
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._model = self._load_model()
        if self.peft_config is not None:
            log.info(
                "HF backend initialized in LoRA mode. Trainable params: %d",
                sum(p.numel() for p in self._model.parameters() if p.requires_grad),
            )

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            seed_everything(seed)
        if self.peft_config is not None:
            _reset_lora(self._model)
        else:
            del self._model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model = self._load_model()

    def fit(
        self,
        examples: list[tuple[str, str]],
        finetune_mode: FinetuneMode,
        *,
        n_steps: int,
        lr: float,
        max_length: int | None = None,
    ) -> list[float]:
        if not examples:
            raise ValueError("examples must contain at least one (goal, target) pair.")

        dataset = self._build_dataset(
            examples=examples,
            finetune_mode=finetune_mode,
            max_length=max_length if max_length is not None else self.max_length,
        )

        train_args = dict(self._train_args)
        train_args.pop("max_steps", None)
        train_args.pop("learning_rate", None)

        return ttt_train(
            self._model,
            dataset,
            n_steps=n_steps,
            lr=lr,
            **train_args,
        )

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
        if (goal is None) == (messages is None):
            raise ValueError("Pass exactly one of goal=… or messages=….")

        self._model.eval()
        if (
            hasattr(self._model, "is_gradient_checkpointing")
            and self._model.is_gradient_checkpointing
        ):
            self._model.gradient_checkpointing_disable()

        prompt = (
            self._build_prompt_text(goal)
            if goal is not None
            else self.apply_chat_template(messages)
        )
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            add_special_tokens=False,
        ).to(self._model.device)
        if self.max_length is not None and inputs.input_ids.shape[1] == self.max_length:
            log.warning(
                f"generate(): prompt hit max_length={self.max_length} tokens "
                "and was right-truncated. TinkerBackend would pass the full "
                "prompt to the server; results may differ across backends."
            )

        gen_kwargs = {
            "num_return_sequences": num_samples,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        decoded = self.tokenizer.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )
        return [d.lstrip(" ") for d in decoded]

    def perplexity(self, dataset, **kwargs) -> dict:
        """Compute per-example perplexity on a HF dataset. Used by the defense module.

        ``kwargs`` are forwarded to :func:`compute_dataset_perplexity`; pass
        ``goal_column``, ``target_column``, ``batch_size``, ``prefix_lengths``
        as needed.
        """
        kwargs.setdefault("system_prompt", self.system_prompt)
        kwargs.setdefault("enable_thinking", self.enable_thinking)
        return compute_dataset_perplexity(
            model=self._model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            **kwargs,
        )

    @property
    def model(self):
        """Access to the underlying HF model (e.g., for defense baseline pass)."""
        return self._model

    def _load_model(self):
        defaults = {
            "device_map": "auto",
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
        }
        kwargs = {**defaults, **self._model_kwargs}
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self.peft_config is not None:
            from peft import get_peft_model

            model = get_peft_model(model, self.peft_config)
        return model

    def _build_prompt_text(self, goal: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": goal})
        return self.apply_chat_template(messages)

    def _build_dataset(
        self,
        *,
        examples: list[tuple[str, str]],
        finetune_mode: FinetuneMode,
        max_length: int | None,
    ):
        if finetune_mode not in _FINETUNE_DEFAULTS:
            raise ValueError(
                f"Unknown finetune_mode: {finetune_mode!r}. "
                f"Expected one of {list(_FINETUNE_DEFAULTS)}."
            )

        finetune_cfg: dict[str, Any] = {
            "name": finetune_mode,
            **_FINETUNE_DEFAULTS[finetune_mode],
        }

        per_example = []
        for goal, target in examples:
            problem = {
                "prompt": self._build_prompt_text(goal),
                "goal": goal,
                "target": target,
            }
            per_example.append(
                prepare_ttt_data(
                    problem=problem,
                    tokenizer=self.tokenizer,
                    max_length=max_length,
                    **finetune_cfg,
                )
            )

        if len(per_example) == 1:
            return per_example[0]

        from datasets import concatenate_datasets

        return concatenate_datasets(per_example)
