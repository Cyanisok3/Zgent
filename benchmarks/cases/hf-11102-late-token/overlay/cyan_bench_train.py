from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from transformers import BertConfig, BertLMHeadModel, BertTokenizerFast, Trainer, TrainerCallback, TrainingArguments

NORMAL_SAMPLES = 20
LOG_EVERY = 1


class LateTokenDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(self, special_token_id: int) -> None:
        self._special_token_id = special_token_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for index in range(NORMAL_SAMPLES + 1):
            token_id = self._special_token_id if index == NORMAL_SAMPLES else 5
            input_ids = torch.tensor([2, 5, 6, token_id, 7, 8, 3, 0])
            yield {
                "input_ids": input_ids,
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 0]),
                "labels": input_ids.clone(),
            }


class StepCallback(TrainerCallback):
    def on_step_end(self, args: object, state: object, control: object, **kwargs: object) -> None:
        del args, control, kwargs
        print(f"training-step={state.global_step}", flush=True)


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
    (directory / "config.json").write_text(json.dumps({"model_type": "bert"}), encoding="utf-8")
    return BertTokenizerFast.from_pretrained(directory, model_max_length=32)


def main() -> None:
    root = Path(__file__).resolve().parent
    tokenizer = tokenizer_at(root)
    model = BertLMHeadModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            is_decoder=True,
        )
    )
    special_token_id = tokenizer.add_special_tokens({"additional_special_tokens": ["[NEW_TOKEN]"]})
    special_token_id = tokenizer.convert_tokens_to_ids("[NEW_TOKEN]")
    model.resize_token_embeddings(len(tokenizer))
    dataset = LateTokenDataset(special_token_id)
    arguments = TrainingArguments(
        output_dir=str(root / "cyan_bench_output"),
        max_steps=NORMAL_SAMPLES + 1,
        per_device_train_batch_size=1,
        logging_steps=LOG_EVERY,
        save_strategy="no",
        report_to=[],
        seed=0,
        use_cpu=True,
        disable_tqdm=True,
    )
    Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        callbacks=[StepCallback()],
    ).train()


if __name__ == "__main__":
    main()
