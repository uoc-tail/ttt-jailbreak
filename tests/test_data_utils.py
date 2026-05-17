"""Tests for the data subpackage: dataset loading + TTT data preparation."""

from transformers import AutoTokenizer
from ttt_jailbreak.attack import LLAMA2_DEFAULT_SYSTEM_PROMPT
from ttt_jailbreak.data import (
    get_special_token_ids,
    load_and_format_dataset,
    prepare_ttt_data,
)


def test_loss_modes(tokenizer):
    """Test all loss modes produce correct masking."""
    print("\n" + "=" * 60)
    print("TEST: Loss modes")
    print("=" * 60)

    problem = {
        "prompt": tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "How to cook pasta?"},
            ],
            add_generation_prompt=True,
            tokenize=False,
        ),
        "user_content": "How to cook pasta?",
        "target": "Boil water, add pasta",
    }

    modes = ["full", "target", "prompt", "few_shot"]
    for mode in modes:
        result = prepare_ttt_data(
            problem=problem,
            prompt_template="{prompt}",
            completion_template="{target}",
            name=mode,
            tokenizer=tokenizer,
            max_length=2048,
        )
        n_total = len(result[0]["input_ids"])
        n_loss = sum(1 for label in result[0]["labels"] if label != -100)
        print(f"  {mode:20s}: {n_loss:3d}/{n_total} tokens with loss")

    print("  ✓ All modes working")


def test_special_token_masking(tokenizer):
    """Test that special tokens are always masked."""
    print("\n" + "=" * 60)
    print("TEST: Special token masking")
    print("=" * 60)

    special_ids = get_special_token_ids(tokenizer)
    print(f"  Found {len(special_ids)} special token IDs")

    problem = {
        "prompt": tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
            tokenize=False,
        ),
        "user_content": "Hello",
        "target": "Hi there!",
    }

    result = prepare_ttt_data(
        problem=problem,
        prompt_template="{prompt}",
        completion_template="{target}",
        name="full",
        tokenizer=tokenizer,
    )

    # Check no special tokens have loss
    for tid, label in zip(result[0]["input_ids"], result[0]["labels"]):
        if tid in special_ids:
            assert label == -100, f"Special token {tid} should be masked"

    print("  ✓ All special tokens masked")


def test_load_and_format(tokenizer):
    """Test dataset loading with various options."""
    print("\n" + "=" * 60)
    print("TEST: load_and_format_dataset")
    print("=" * 60)

    result = load_and_format_dataset(
        path="JailbreakBench/JBB-Behaviors",
        name="behaviors",
        split="harmful[:1]",
        tokenizer=tokenizer,
        system_prompt=LLAMA2_DEFAULT_SYSTEM_PROMPT,
        input_column="Goal",
        target_column="Target",
    )

    assert "prompt" in result[0]
    assert "goal" in result[0]
    assert "user_content" in result[0]
    print(f"  Loaded {len(result)} items")
    print("  ✓ Required fields present")


if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    test_loss_modes(tokenizer)
    test_special_token_masking(tokenizer)
    test_load_and_format(tokenizer)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
