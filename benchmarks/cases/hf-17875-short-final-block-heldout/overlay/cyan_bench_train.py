from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

BLOCK_SIZE = 8
FULL_BLOCKS = 20
LOG_EVERY = 10


def group_tokens(tokens: list[int]) -> list[list[int]]:
    total_length = len(tokens)
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    return [tokens[index : index + BLOCK_SIZE] for index in range(0, total_length, BLOCK_SIZE)]


class GroupedLanguageModelDataset(IterableDataset):
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        records = [[1, 15, 16, 17, 18, 19, 20, 2] for _ in range(FULL_BLOCKS)]
        records.append([1, 21, 22, 23, 24, 25, 2])
        for record in records:
            for block in group_tokens(record):
                tensor = torch.tensor(block)
                yield {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                    "labels": tensor.clone(),
                }


class StepLogCallback(TrainerCallback):
    def on_log(self, args: object, state: object, control: object, **kwargs: object) -> None:
        del args, control, kwargs
        print(f"training-step={state.global_step}", flush=True)


torch.manual_seed(41)
model = BertForMaskedLM(
    BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
)
steps_per_epoch = (FULL_BLOCKS + 1) // 2
arguments = TrainingArguments(
    output_dir=str(Path("outputs")),
    max_steps=steps_per_epoch + 1,
    per_device_train_batch_size=2,
    disable_tqdm=True,
    report_to=[],
    save_strategy="no",
    logging_steps=LOG_EVERY,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=GroupedLanguageModelDataset(),
    callbacks=[StepLogCallback()],
)
trainer.train()
print("training-complete", flush=True)
