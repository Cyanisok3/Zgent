from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


# 写入完全本地的 tiny BERT、WordPiece 词表和训练文本
def prepare_assets(root: Path) -> tuple[Path, Path, Path]:
    model_dir = root / "cyan_bench_assets" / "model"
    tokenizer_dir = root / "cyan_bench_assets" / "tokenizer"
    data_dir = root / "cyan_bench_assets" / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["BertLMHeadModel"],
        "attention_probs_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 32,
        "intermediate_size": 64,
        "is_decoder": True,
        "max_position_embeddings": 64,
        "model_type": "bert",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 32,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "training",
        "model",
        "data",
        "sample",
        "token",
        "stream",
        "batch",
        "step",
        "local",
        "small",
        "text",
        "learns",
        "from",
        "the",
        "sequence",
        ".",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "end",
    ]
    (tokenizer_dir / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps({"do_lower_case": True, "model_max_length": 64}), encoding="utf-8"
    )
    (tokenizer_dir / "config.json").write_text(
        json.dumps({"model_type": "bert"}), encoding="utf-8"
    )
    (tokenizer_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
                "sep_token": "[SEP]",
                "unk_token": "[UNK]",
            }
        ),
        encoding="utf-8",
    )
    sentence = "training model learns from the local data sequence one two three four five."
    train_file = data_dir / "train.txt"
    train_file.write_text("\n".join(sentence for _ in range(200)) + "\n", encoding="utf-8")
    return model_dir, tokenizer_dir, train_file


# 以官方 run_clm 入口执行两个真实 Trainer step
def main() -> None:
    root = Path(__file__).resolve().parent
    model_dir, tokenizer_dir, train_file = prepare_assets(root)
    output_dir = root / "cyan_bench_output"
    script = root / "examples" / "pytorch" / "language-modeling" / "run_clm.py"
    sys.argv = [
        str(script),
        "--train_file",
        str(train_file),
        "--streaming",
        "--validation_split_percentage",
        "20",
        "--config_name",
        str(model_dir),
        "--tokenizer_name",
        str(tokenizer_dir),
        "--do_train",
        "--output_dir",
        str(output_dir),
        "--overwrite_output_dir",
        "--max_steps",
        "2",
        "--per_device_train_batch_size",
        "2",
        "--block_size",
        "16",
        "--logging_steps",
        "1",
        "--save_strategy",
        "no",
        "--report_to",
        "none",
        "--seed",
        "0",
    ]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
