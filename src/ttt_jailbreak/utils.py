"""Truly miscellaneous helpers that don't belong to a specific subpackage."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
from transformers import set_seed as hf_set_seed

log = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA), and HuggingFace for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    log.info(f"Set all random seeds to {seed}")
