# Contributing

## Adding a new TTT backend

Backends implement the abstract base class `ttt_jailbreak.ttt.TTTBackend`. To
add one:

1. **Create the implementation.** Put it at
   `src/ttt_jailbreak/ttt/<your_backend>.py`. Subclass `TTTBackend` and
   implement four methods:

   | Method | Contract |
   |---|---|
   | `__init__(model_name, *, system_prompt, max_length, **kwargs)` | Fix everything that does not change across problems. Load tokenizer / renderer / client. |
   | `fit(examples, finetune_mode, *, n_steps, lr, **kwargs) -> list[float]` | Train on a list of `(goal, target)` pairs. Return per-step losses. |
   | `generate(goal=None, *, messages=None, num_samples, max_new_tokens, …) -> list[str]` | Sample `num_samples` completions. Accept either a raw goal (single-turn) or an explicit messages list (multi-turn / ICL). |
   | `reset(seed=None) -> None` | Restore the model to its pre-fit state. Applied between problems. |

   If the backend can compute per-example perplexity (used by the defense
   module), set `supports_perplexity = True` and implement `perplexity()`.

2. **Register a Hydra config.** Add `conf/backend/<your_backend>.yaml` with a
   `_target_` pointing at your class. The existing `conf/backend/hf.yaml` is
   a good reference; it forwards values from `cfg.model`, `cfg.data`, and
   `cfg.trainer` so the backend can be selected with a single CLI flag
   (`backend=<your_backend>`).

3. **Optionally export from the package.** Add an entry to
   `src/ttt_jailbreak/ttt/__init__.py` so callers can `from ttt_jailbreak.ttt
   import YourBackend`. Use lazy imports (see how `TinkerBackend` is wired
   via `__getattr__`) if your backend pulls in heavy optional dependencies.

4. **Add a smoke test.** Mirror `tests/test_ttt_backend.py`: a small fit →
   generate → reset cycle on a tiny model. Skip the test cleanly when the
   backend's prerequisites are missing (`pytest.skip(...)`).

## Style

- One logical change per commit; commit subject only, no body.
- Match the existing code style. Keep diffs focused on the requested change.
- If you remove a dependency, run `uv sync` so `pyproject.toml` and `uv.lock`
  stay in sync.
