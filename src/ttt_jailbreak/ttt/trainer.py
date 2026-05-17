"""TTT inner training loop.

Single function that drives ``n_steps`` of gradient descent on a pre-tokenized
dataset. Designed to be called by :class:`HuggingFaceBackend.fit`; usable
standalone for tests.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch


def ttt_train(
    model,
    dataset,
    *,
    n_steps: int,
    lr: float,
    optimizer: str = "adamw",
    grad_clip: float | None = None,
    weight_decay: float = 0.0,
    bf16: bool = False,
    lr_scheduler: str = "constant",
    gradient_checkpointing: bool = False,
) -> list[float]:
    """Train ``model`` on a pre-tokenized HF ``dataset``. Returns per-step losses."""
    device = next(model.parameters()).device
    n_examples = len(dataset)

    all_input_ids = []
    all_labels = []
    for i in range(n_examples):
        ex = dataset[i]
        all_input_ids.append(
            torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
        )
        all_labels.append(torch.tensor([ex["labels"]], dtype=torch.long, device=device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer == "adamw":
        opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Use 'adamw' or 'sgd'.")

    if lr_scheduler == "constant":
        scheduler = None
    elif lr_scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: max(0.0, 1.0 - step / n_steps)
        )
    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    else:
        raise ValueError(
            f"Unknown lr_scheduler: {lr_scheduler!r}. "
            "Use 'constant', 'linear', or 'cosine'."
        )

    autocast = (
        torch.amp.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()
    losses: list[float] = []

    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0

        for i in range(n_examples):
            with autocast:
                outputs = model(input_ids=all_input_ids[i], labels=all_labels[i])
            loss = outputs.loss / n_examples
            loss.backward()
            step_loss += loss.item()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        opt.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(step_loss)

    return losses
