"""Tinker (https://tinker-docs.thinkingmachines.ai/) backend for TTT.

Uses the Tinker API for LoRA fine-tuning and sampling. Tokenization and data
preparation go through ``tinker_cookbook``'s renderer to match Tinker's hosted
models. Generation supports reasoning-style outputs (analysis/final channels
for GPT-OSS) by splitting the raw decoded text into ``reasoning`` and ``text``
parts at sample time.

The backend exposes a synchronous interface to callers; internally each fit /
generate call runs a short asyncio loop. Cross-problem concurrency is the
caller's responsibility (e.g., SLURM array sharding).

Prerequisites:
    export TINKER_API_KEY=<your-key>
    uv add tinker tinker-cookbook
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import torch

from .base import FinetuneMode, TTTBackend

log = logging.getLogger(__name__)


# Map model name → renderer name. The renderer controls how messages get
# tokenized into Tinker's loss-weighted training format. Extend this map for
# new Tinker-hosted models.
DEFAULT_RENDERERS: dict[str, str] = {
    "Qwen/Qwen3-8B": "qwen3_disable_thinking",
    "Qwen/Qwen3-32B": "qwen3_disable_thinking",
    "Qwen/Qwen3-8B-Base": "qwen3",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3",
    "meta-llama/Llama-3.1-8B": "llama3",
    "meta-llama/Llama-3.1-70B": "llama3",
    "meta-llama/Llama-3.3-70B-Instruct": "llama3",
    "openai/gpt-oss-120b": "gpt_oss_no_sysprompt",
    "openai/gpt-oss-20b": "gpt_oss_no_sysprompt",
}


def _resolve_renderer(model_name: str, override: str | None) -> str:
    if override:
        return override
    if model_name in DEFAULT_RENDERERS:
        return DEFAULT_RENDERERS[model_name]
    from tinker_cookbook.model_info import get_recommended_renderer_names

    recommended = get_recommended_renderer_names(model_name)
    if recommended:
        return recommended[0]
    raise ValueError(
        f"No renderer registered for {model_name!r}. "
        f"Pass renderer_name=… or extend DEFAULT_RENDERERS in tinker.py."
    )


def _strip_special_token_weights(model_input, weights, tokenizer):
    """Zero loss weights on EOS / BOS / pad tokens so the model does not learn
    to stop generation right after the target string. Mirrors the HF backend's
    ``mask_special_tokens=True`` behavior in prepare_ttt_data."""
    tokens = []
    for chunk in model_input.chunks:
        if hasattr(chunk, "tokens"):
            tokens.extend(chunk.tokens)
        else:
            tokens.extend([0] * chunk.length)

    special_ids = set()
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)

    for i, t in enumerate(tokens):
        if i < len(weights) and t in special_ids:
            weights[i] = 0.0
    return weights


class TinkerBackend(TTTBackend):
    """Tinker API backend.

    Per-problem flow: ``reset()`` creates a fresh LoRA training client, ``fit()``
    runs ``n_steps`` of forward/backward + optim_step, then snapshots a sampling
    client for ``generate()``.
    """

    supports_perplexity = False

    def __init__(
        self,
        model_name: str,
        *,
        system_prompt: str = "",
        max_length: int | None = None,
        renderer_name: str | None = None,
        lora_rank: int = 32,
        service_client_kwargs: dict | None = None,
    ):
        import tinker
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.lora_rank = lora_rank

        self.tokenizer = get_tokenizer(model_name)
        renderer_name = _resolve_renderer(model_name, renderer_name)
        self.renderer = get_renderer(renderer_name, self.tokenizer)
        self.renderer_name = renderer_name
        log.info(f"Tinker renderer: {renderer_name}")

        self._service_client = tinker.ServiceClient(**(service_client_kwargs or {}))
        self._training_client = None
        self._sampling_client = None

    def reset(self, seed: int | None = None) -> None:
        # ``seed`` is accepted for API parity but is not used: Tinker controls
        # adapter initialization server-side, and each new training client gets
        # a fresh adapter.
        del seed
        self._training_client = asyncio.run(
            self._service_client.create_lora_training_client_async(
                base_model=self.model_name,
                rank=self.lora_rank,
            )
        )
        self._sampling_client = None

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
        if self._training_client is None:
            self.reset()

        data = self._build_tinker_data(
            examples=examples,
            finetune_mode=finetune_mode,
            max_length=max_length if max_length is not None else self.max_length,
        )
        losses = asyncio.run(self._fit_async(data, n_steps=n_steps, lr=lr))
        self._sampling_client = None  # invalidate, next generate() snapshots fresh
        return losses

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

        if self._sampling_client is None:
            if self._training_client is None:
                self.reset()
            self._sampling_client = asyncio.run(
                self._training_client.save_weights_and_get_sampling_client_async()
            )

        import tinker

        if messages is None:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": goal})
        gen_prompt = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()

        params = tinker.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            stop=stop or None,
        )
        result = self._sampling_client.sample(
            prompt=gen_prompt,
            num_samples=num_samples,
            sampling_params=params,
        ).result()

        is_reasoning_model = self.renderer_name.startswith("gpt_oss")
        generations = []
        for seq in result.sequences:
            raw = self.tokenizer.decode(seq.tokens, skip_special_tokens=True).lstrip(
                " "
            )
            generations.append(_extract_final_text(raw) if is_reasoning_model else raw)
        return generations

    async def _fit_async(self, data, *, n_steps: int, lr: float) -> list[float]:
        import tinker

        adam_params = tinker.AdamParams(learning_rate=lr)
        losses: list[float] = []
        for _ in range(n_steps):
            fwd_bwd_future = await self._training_client.forward_backward_async(
                data, loss_fn="cross_entropy"
            )
            optim_future = await self._training_client.optim_step_async(adam_params)
            fwd_bwd_result = await fwd_bwd_future.result_async()
            await optim_future.result_async()
            losses.append(_average_nll(fwd_bwd_result, data))
        return losses

    def _build_tinker_data(
        self,
        *,
        examples: list[tuple[str, str]],
        finetune_mode: FinetuneMode,
        max_length: int | None,
    ):
        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        if not examples:
            raise ValueError("examples must contain at least one (goal, target) pair.")

        data = []
        for goal, target in examples:
            messages: list[dict[str, Any]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            if finetune_mode == "prompt":
                messages.append({"role": "user", "content": goal + target})
                model_input, weights = self.renderer.build_supervised_example(messages)
                weights = torch.ones_like(weights)
            elif finetune_mode == "target":
                messages.append({"role": "user", "content": goal})
                messages.append({"role": "assistant", "content": target})
                model_input, weights = self.renderer.build_supervised_example(messages)
            elif finetune_mode == "few_shot":
                messages.append({"role": "user", "content": goal})
                messages.append({"role": "assistant", "content": target})
                model_input, weights = self.renderer.build_supervised_example(messages)
                weights = torch.ones_like(weights)
            else:
                raise ValueError(f"Unknown finetune_mode: {finetune_mode!r}")

            weights = _strip_special_token_weights(model_input, weights, self.tokenizer)
            data.append(
                datum_from_model_input_weights(
                    model_input, weights, max_length=max_length
                )
            )
        return data


def _average_nll(fwd_bwd_result, data) -> float:
    logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
    weights_list = [d.loss_fn_inputs["weights"] for d in data]
    total_w = 0.0
    total_wl = 0.0
    for lp, w in zip(logprobs, weights_list):
        lp_t = lp.to_torch() if hasattr(lp, "to_torch") else torch.tensor(lp.data)
        w_t = w.to_torch() if hasattr(w, "to_torch") else torch.tensor(w.data)
        total_wl += lp_t.dot(w_t).item()
        total_w += w_t.sum().item()
    return -total_wl / total_w if total_w > 0 else float("nan")


def _extract_final_text(raw: str) -> str:
    """Strip GPT-OSS reasoning channel.

    GPT-OSS outputs ``analysis<thoughts>final<answer>``. We return the
    post-``final`` slice. Only call this when the renderer is known to be a
    GPT-OSS variant; otherwise legitimate text containing the word "final"
    gets truncated.
    """
    if "final" in raw:
        return raw.split("final", 1)[1].strip()
    return raw
