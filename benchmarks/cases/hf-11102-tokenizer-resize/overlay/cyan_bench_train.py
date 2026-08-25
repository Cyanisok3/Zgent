from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertLMHeadModel, BertTokenizerFast, Trainer, TrainingArguments


class TokenDataset(Dataset[dict[str, torch.Tensor]]):
    # 保存已经 tokenized 的固定训练样本
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self._samples = samples

    # 返回训练样本数量
    def __len__(self) -> int:
        return len(self._samples)

    # 返回带 language-model labels 的单个样本
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self._samples[0].items()}
        item["labels"] = item["input_ids"].clone()
        return item


# 创建可离线训练的最小 WordPiece tokenizer
def tokenizer_at(root: Path) -> BertTokenizerFast:
    directory = root / "cyan_bench_tokenizer"
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "training",
        "model",
        "learns",
        "new",
        "token",
        "sequence",
        ".",
    ]
    (directory / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    (directory / "config.json").write_text(
        json.dumps({"model_type": "bert"}), encoding="utf-8"
    )
    return BertTokenizerFast.from_pretrained(directory, model_max_length=32)


# 运行包含新增 special token 的真实 Trainer step
def main() -> None:
    root = Path(__file__).resolve().parent
    tokenizer = tokenizer_at(root)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
        is_decoder=True,
    )
    model = BertLMHeadModel(config)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[NEW_TOKEN]"]})
    model.resize_token_embeddings(len(tokenizer))
    encoded = tokenizer(
        ["training model learns [NEW_TOKEN] ."] * 4,
        padding="max_length",
        truncation=True,
        max_length=12,
        return_tensors="pt",
    )
    dataset = TokenDataset([encoded])
    arguments = TrainingArguments(
        output_dir=str(root / "cyan_bench_output"),
        max_steps=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=0,
        use_cpu=True,
    )
    Trainer(model=model, args=arguments, train_dataset=dataset).train()


if __name__ == "__main__":
    main()
