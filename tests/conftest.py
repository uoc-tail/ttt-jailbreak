import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def tokenizer():
    """Shared tokenizer fixture for data_utils tests."""
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
    )
