"""vLLM server helpers.

The judging module uses these to send batched chat-completion requests to a
running vLLM server, and to fall back to in-process vLLM if the server is
offline. Functions here are intentionally thin wrappers around ``requests``
and the vLLM SDK.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def check_server_health(base_url: str, timeout: float = 5.0) -> bool:
    """Return True if the vLLM server's /models endpoint responds with 200."""
    try:
        response = requests.get(f"{base_url}/models", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException as e:
        log.debug(f"Server health check failed: {e}")
        return False


def run_chat_completion(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 120.0,
    seed: int | None = None,
) -> dict:
    """Send a single chat completion. Returns ``{content, reasoning_content}``."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        return {
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
        }
    except requests.RequestException as e:
        log.error(f"Chat completion request failed: {e}")
        return {"content": "", "reasoning_content": "", "error": f"request: {e}"}
    except (KeyError, IndexError) as e:
        log.error(f"Unexpected response format: {e}")
        return {"content": "", "reasoning_content": "", "error": f"format: {e}"}


def run_batch_chat_completions(
    base_url: str,
    model_name: str,
    prompts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 256,
    batch_size: int = 50,
    timeout: float = 120.0,
    verbose: bool = True,
    seed: int | None = None,
) -> List[dict]:
    """Batched ``run_chat_completion`` with a ThreadPoolExecutor."""
    generated_results: List[dict] = [None] * len(prompts)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=min(batch_size, 32)) as executor:
            futures = {}
            for j, prompt in enumerate(batch_prompts):
                future = executor.submit(
                    run_chat_completion,
                    base_url=base_url,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    seed=seed,
                )
                futures[future] = i + j
            for future in as_completed(futures):
                idx = futures[future]
                generated_results[idx] = future.result()

        if verbose and i + batch_size < len(prompts):
            log.info(f"Processed {i + batch_size}/{len(prompts)} prompts...")

    return generated_results


def build_judge_prompt(system_prompt: str, goal: str, response: str) -> str:
    """Substitute ``{goal}`` / ``{response}`` placeholders in the judge template."""
    return system_prompt.format(goal=goal, response=response)


def run_local_inference(cfg: DictConfig, prompts: list) -> list[dict]:
    """Fall back to in-process vLLM when no server is available.

    Returns one ``{content, reasoning_content, error}`` dict per prompt, matching
    the shape ``run_chat_completion`` returns over the network. The model is
    loaded from ``cfg.judge.model.*`` and the chat template is applied before
    sampling.
    """
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError:
        log.error(
            "vllm and transformers are required for local inference. "
            "Install them or start a vLLM server."
        )
        raise

    log.info(f"Loading local model: {cfg.judge.model.path}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.judge.model.path)

    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    llm = LLM(
        model=cfg.judge.model.path,
        dtype=cfg.judge.model.dtype,
        tensor_parallel_size=cfg.judge.model.tensor_parallel_size,
        gpu_memory_utilization=cfg.judge.model.gpu_memory_utilization,
        max_model_len=cfg.judge.model.max_model_len,
        quantization=cfg.judge.model.get("quantization", None),
    )
    sampling_params = SamplingParams(
        temperature=cfg.judge.sampling.temperature,
        max_tokens=cfg.judge.sampling.max_tokens,
    )
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [
        {"content": o.outputs[0].text, "reasoning_content": "", "error": None}
        for o in outputs
    ]
