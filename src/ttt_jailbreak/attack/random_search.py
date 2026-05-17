"""Loss-guided random search over adversarial suffixes.

Token-level RS that mutates a fixed-length suffix and keeps any candidate that
lowers cross-entropy on the first target token. Mirrors the optimization loop
from Andriushchenko et al. (2024).
"""

from __future__ import annotations

import logging
import math
import random
from typing import List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


_MARKER = "<<<SUFFIX_MARKER>>>"


def get_disallowed_token_ids(
    tokenizer,
    allow_non_ascii: bool = False,
) -> Set[int]:
    """Return token IDs that should never appear in an adversarial suffix.

    Filters: BOS/EOS/PAD/UNK, special tokens that vanish under
    ``skip_special_tokens=True``, and (by default) non-ASCII/non-printable
    tokens. Matches AdversariaLLM (Zou et al., 2023).
    """
    disallowed: Set[int] = set()
    n_tokens = len(tokenizer)

    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            disallowed.add(token_id)

    for i in range(n_tokens):
        try:
            if not tokenizer.decode([i], skip_special_tokens=True):
                disallowed.add(i)
        except Exception:
            disallowed.add(i)

    if not allow_non_ascii:
        for i in range(n_tokens):
            if i in disallowed:
                continue
            try:
                decoded = tokenizer.decode([i])
                if not (decoded.isascii() and decoded.isprintable()):
                    disallowed.add(i)
            except Exception:
                disallowed.add(i)

    return disallowed


def mutate_suffix_tokens(
    tokens: List[int],
    n_changes: int,
    vocab_size: int = 0,
    allowed_tokens: Optional[List[int]] = None,
) -> List[int]:
    """Replace ``n_changes`` random positions with tokens drawn from ``allowed_tokens``."""
    if not tokens:
        if allowed_tokens is not None:
            return [random.choice(allowed_tokens) for _ in range(n_changes)]
        return np.random.randint(0, vocab_size, n_changes).tolist()

    tokens = list(tokens)
    n_changes = min(n_changes, len(tokens))
    positions = random.sample(range(len(tokens)), n_changes)
    for pos in positions:
        if allowed_tokens is not None:
            tokens[pos] = random.choice(allowed_tokens)
        else:
            tokens[pos] = random.randint(0, vocab_size - 1)
    return tokens


def _apply_template(
    tokenizer,
    messages,
    system_prompt,
    add_generation_prompt: bool = False,
    enable_thinking: Optional[bool] = None,
) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(messages)
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    text = tokenizer.apply_chat_template(msgs, **kwargs)
    # Strip BOS if the template prepends it (re-added by the tokenizer).
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token) :]
    return text


def _tokenize(tokenizer, text: str) -> list[int]:
    return (
        tokenizer(text, add_special_tokens=True, return_tensors="pt")
        .input_ids[0]
        .tolist()
    )


def get_template_parts(
    tokenizer,
    user_content: str,
    target_str: str,
    system_prompt: Optional[str],
    enable_thinking: Optional[bool] = None,
):
    """Split a chat-templated conversation at the suffix-insertion point.

    Returns ``(before_ids, after_ids, target_offset)`` where
    ``before_ids + suffix_ids + after_ids`` reconstructs the full input and
    ``target_offset`` is the number of post-suffix template tokens preceding
    the target string.
    """
    text_full = _apply_template(
        tokenizer,
        [
            {"role": "user", "content": user_content + _MARKER},
            {"role": "assistant", "content": target_str},
        ],
        system_prompt,
        enable_thinking=enable_thinking,
    )

    parts = text_full.split(_MARKER)
    assert len(parts) == 2, f"Marker split gave {len(parts)} parts (expected 2)"
    before_ids = _tokenize(tokenizer, parts[0])
    after_ids = tokenizer.encode(parts[1], add_special_tokens=False)

    text_gen = _apply_template(
        tokenizer,
        [{"role": "user", "content": user_content + _MARKER}],
        system_prompt,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    parts_gen = text_gen.split(_MARKER)
    gen_after_ids = tokenizer.encode(parts_gen[1], add_special_tokens=False)
    target_offset = len(gen_after_ids)

    return before_ids, after_ids, target_offset


def compute_target_loss(model, input_ids, target_mask) -> torch.Tensor:
    """CE loss on the first target token per row (Andriushchenko et al. 2404.02151)."""
    with torch.no_grad():
        logits = model(input_ids).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = target_mask[:, 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_logits.size(0), -1)

    first_target_idx = shift_mask.float().argmax(dim=1)
    return loss_per_token.gather(1, first_target_idx.unsqueeze(1)).squeeze(1)


def build_input_ids_and_mask(
    before_ids: list[int],
    suffix_ids: list[int],
    after_ids: list[int],
    target_offset: int,
    device,
):
    """Assemble ``(input_ids, target_mask)`` tensors for a single candidate."""
    all_ids = list(before_ids) + list(suffix_ids) + list(after_ids)
    seq_len = len(all_ids)
    n_prefix = len(before_ids) + len(suffix_ids) + target_offset

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    target_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    target_mask[0, n_prefix:] = True

    return input_ids, target_mask


def _batched_forward(model, batch_input_ids, batch_target_mask, max_batch_size):
    """Forward pass with automatic sub-batching on OOM."""
    batch_size = min(max_batch_size, batch_input_ids.size(0))

    while batch_size >= 1:
        try:
            all_losses = []
            for i in range(0, batch_input_ids.size(0), batch_size):
                sub_ids = batch_input_ids[i : i + batch_size]
                sub_mask = batch_target_mask[i : i + batch_size]
                losses = compute_target_loss(model, sub_ids, sub_mask)
                all_losses.append(losses)
            return torch.cat(all_losses)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = batch_size // 2
            if batch_size < 1:
                raise
            log.warning(f"OOM, reducing batch size to {batch_size}")


def _build_allowed_tokens(tokenizer, attack_cfg) -> list[int]:
    disallowed = get_disallowed_token_ids(
        tokenizer,
        allow_non_ascii=attack_cfg.allow_non_ascii,
    )
    all_token_ids = set(range(tokenizer.vocab_size))
    allowed_ids = sorted(all_token_ids - disallowed)
    log.info(f"Allowed tokens: {len(allowed_ids)}/{tokenizer.vocab_size}")
    return allowed_ids


def _init_suffix(tokenizer, attack_cfg, allowed_ids) -> list[int]:
    n_tokens = attack_cfg.n_tokens_adv
    adv_init = attack_cfg.get("adv_init", "")
    if adv_init and adv_init.strip():
        suffix_ids = tokenizer.encode(adv_init, add_special_tokens=False)
        if len(suffix_ids) < n_tokens:
            suffix_ids += [
                random.choice(allowed_ids) for _ in range(n_tokens - len(suffix_ids))
            ]
        elif len(suffix_ids) > n_tokens:
            suffix_ids = suffix_ids[:n_tokens]
    else:
        suffix_ids = [random.choice(allowed_ids) for _ in range(n_tokens)]
    return suffix_ids


def run_random_search(model, tokenizer, before_ids, after_ids, target_offset, cfg):
    """Optimize the suffix to minimize CE on the first target token.

    Returns ``(best_suffix_ids, best_loss, loss_history, token_prob_history)``.
    """
    attack_cfg = cfg.attack
    num_steps = attack_cfg.num_steps
    candidates_per_step = attack_cfg.candidates_per_generation
    radius = attack_cfg.neighborhood_radius
    device = next(model.parameters()).device

    allowed_ids = _build_allowed_tokens(tokenizer, attack_cfg)
    suffix_ids = _init_suffix(tokenizer, attack_cfg, allowed_ids)

    input_ids, target_mask = build_input_ids_and_mask(
        before_ids,
        suffix_ids,
        after_ids,
        target_offset,
        device,
    )
    best_loss = compute_target_loss(model, input_ids, target_mask).item()
    best_suffix_ids = list(suffix_ids)
    loss_history = [best_loss]
    token_prob_history = [math.exp(-best_loss)]

    log.info(f"Initial loss: {best_loss:.4f}")
    early_stop_prob = attack_cfg.get("early_stop_prob", 0)

    for step in range(num_steps):
        candidates = [
            mutate_suffix_tokens(
                list(best_suffix_ids),
                n_changes=radius,
                allowed_tokens=allowed_ids,
            )
            for _ in range(candidates_per_step)
        ]

        batch_input_ids = []
        batch_target_mask = []
        for cand in candidates:
            ids, mask = build_input_ids_and_mask(
                before_ids,
                cand,
                after_ids,
                target_offset,
                device,
            )
            batch_input_ids.append(ids)
            batch_target_mask.append(mask)
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_target_mask = torch.cat(batch_target_mask, dim=0)

        losses = _batched_forward(
            model,
            batch_input_ids,
            batch_target_mask,
            candidates_per_step,
        )
        min_idx = losses.argmin().item()
        min_loss = losses[min_idx].item()

        if min_loss < best_loss:
            best_loss = min_loss
            best_suffix_ids = list(candidates[min_idx])

        loss_history.append(best_loss)
        best_prob = math.exp(-best_loss)
        token_prob_history.append(best_prob)

        if step % 10 == 0 or step == num_steps - 1:
            suffix_text = tokenizer.decode(best_suffix_ids)
            log.info(
                f"Step {step}/{num_steps}: loss={best_loss:.4f} "
                f"prob={best_prob:.6f} suffix='{suffix_text[:60]}...'"
            )

        if early_stop_prob > 0 and best_prob >= early_stop_prob:
            log.info(
                f"Early stopping at step {step}: P(target)={best_prob:.4f} "
                f">= {early_stop_prob}"
            )
            break

    return best_suffix_ids, best_loss, loss_history, token_prob_history


def run_random_search_on_backend(
    backend,
    user_content: str,
    target_str: str,
    *,
    target_prefix: str = "",
    system_prompt: str = "",
    cfg,
) -> tuple[str, float]:
    """Public entry point for scripts: prepares the template parts and runs RS
    against the underlying HF model.

    Uses ``backend.enable_thinking`` (if set) so the RS-optimized prompt and
    the generation-time prompt share the same chat-template surface; otherwise
    Qwen3-style backends would optimize over the thinking-enabled rendering
    and generate against the thinking-disabled one.

    Returns ``(suffix_text, best_loss)``.
    """
    loss_target = target_prefix + target_str
    before_ids, after_ids, target_offset = get_template_parts(
        backend.tokenizer,
        user_content,
        loss_target,
        system_prompt,
        enable_thinking=getattr(backend, "enable_thinking", None),
    )
    best_suffix_ids, best_loss, *_ = run_random_search(
        backend.model,
        backend.tokenizer,
        before_ids,
        after_ids,
        target_offset,
        cfg,
    )
    return backend.tokenizer.decode(best_suffix_ids), best_loss
