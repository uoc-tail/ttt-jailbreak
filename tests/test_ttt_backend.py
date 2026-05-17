"""Smoke tests for the TTT backend API.

Each test exercises the public Protocol (fit → generate → reset) on a real
backend. The HF test runs on a small (~135M) instruct model so it is feasible
locally; the Tinker test is skipped without ``TINKER_API_KEY``.
"""

from __future__ import annotations

import os

import pytest
from ttt_jailbreak.ttt import TTTBackend

HF_SMOKE_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"


@pytest.fixture(scope="module")
def hf_backend():
    """HuggingFace backend on a tiny instruct model. Skips on import / load failure."""
    try:
        from ttt_jailbreak.ttt import HuggingFaceBackend
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers stack unavailable: {e}")

    try:
        return HuggingFaceBackend(
            model_name=HF_SMOKE_MODEL,
            system_prompt="You are a helpful assistant.",
            max_length=256,
            # Tiny LoRA so the test exercises the persistent-model path.
            peft_config=_tiny_lora(),
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"could not load {HF_SMOKE_MODEL}: {e}")


def _tiny_lora():
    from peft import LoraConfig

    return LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )


def test_hf_backend_is_a_ttt_backend(hf_backend):
    assert isinstance(hf_backend, TTTBackend)
    assert hf_backend.supports_perplexity is True


def test_hf_generate_returns_non_empty(hf_backend):
    out = hf_backend.generate(
        goal="What is 2 + 2?",
        num_samples=2,
        max_new_tokens=8,
        do_sample=False,
    )
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)


def test_hf_fit_returns_losses(hf_backend):
    losses = hf_backend.fit(
        examples=[("Write a haiku.", "An old silent pond...")],
        finetune_mode="target",
        n_steps=1,
        lr=1e-4,
    )
    assert isinstance(losses, list) and len(losses) == 1
    assert all(isinstance(loss, float) for loss in losses)


def test_hf_reset_restores_base(hf_backend):
    """After reset(), LoRA contribution must be zero again."""
    import torch

    # Run a fit to perturb the adapter.
    hf_backend.fit(
        examples=[("foo", "bar baz")],
        finetune_mode="target",
        n_steps=1,
        lr=1e-3,
    )

    hf_backend.reset(seed=42)

    # After reset, lora_B is zeros → adapter contribution is zero.
    for _, module in hf_backend.model.named_modules():
        if hasattr(module, "lora_B"):
            for key in module.lora_B:
                w = module.lora_B[key].weight
                assert torch.all(w == 0), "lora_B should be re-zeroed after reset()"


def test_generate_rejects_both_or_neither(hf_backend):
    with pytest.raises(ValueError):
        hf_backend.generate(num_samples=1, max_new_tokens=4)
    with pytest.raises(ValueError):
        hf_backend.generate(
            goal="x",
            messages=[{"role": "user", "content": "x"}],
            num_samples=1,
            max_new_tokens=4,
        )


@pytest.mark.skipif(
    "TINKER_API_KEY" not in os.environ,
    reason="TINKER_API_KEY not set; skipping Tinker smoke test.",
)
def test_tinker_backend_smoke():
    from ttt_jailbreak.ttt import TinkerBackend

    backend = TinkerBackend(
        model_name="Qwen/Qwen3-8B",
        system_prompt="",
        max_length=256,
        lora_rank=8,
    )
    assert isinstance(backend, TTTBackend)
    assert backend.supports_perplexity is False

    backend.reset()
    losses = backend.fit(
        examples=[("Write a haiku.", "An old silent pond...")],
        finetune_mode="target",
        n_steps=1,
        lr=5e-5,
    )
    assert len(losses) == 1

    out = backend.generate(
        goal="What is 2 + 2?",
        num_samples=1,
        max_new_tokens=8,
        do_sample=False,
    )
    assert isinstance(out, list) and len(out) == 1
    assert isinstance(out[0], str)
