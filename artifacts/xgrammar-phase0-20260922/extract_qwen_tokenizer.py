#!/usr/bin/env python3
"""Prepare exact Qwen tokenizer fixtures for the standalone XGrammar C++ gate."""

import json
import sys
from pathlib import Path

from transformers import Qwen2TokenizerFast


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_qwen_tokenizer.py MODEL_DIR OUTPUT_DIR")

    model_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_dir, local_files_only=True, fix_mistral_regex=True
    )
    config = json.loads((model_dir / "config.json").read_text())
    model_vocab_size = int(config["vocab_size"])

    vocab = [""] * model_vocab_size
    for token, token_id in tokenizer.get_vocab().items():
        if token_id < model_vocab_size:
            vocab[token_id] = token

    fixtures = {
        "valid_primary": '{"name":"Ada","age":37,"tags":["math","é"]}',
        "valid_secondary": '{"status":"ready","items":[1,2,3]}',
        "invalid_primary": '{"name":7,"age":"wrong","tags":[]}',
    }
    token_ids = {
        name: tokenizer.encode(text, add_special_tokens=False)
        for name, text in fixtures.items()
    }

    (output_dir / "encoded_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "backend_tokenizer.json").write_text(
        tokenizer.backend_tokenizer.to_str(), encoding="utf-8"
    )
    (output_dir / "fixture_tokens.json").write_text(
        json.dumps(token_ids), encoding="utf-8"
    )
    (output_dir / "fixture_text.json").write_text(
        json.dumps(fixtures, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_len": len(tokenizer),
        "model_vocab_size": model_vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "special_token_ids": tokenizer.all_special_ids,
        "fixture_token_counts": {k: len(v) for k, v in token_ids.items()},
    }
    (output_dir / "tokenizer_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
