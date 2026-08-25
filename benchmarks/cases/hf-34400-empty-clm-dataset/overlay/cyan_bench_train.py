from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments

BLOCK_SIZE = 16


def group_tokens(tokens: list[int]) -> list[list[int]]:
    total_length = (len(tokens) // BLOCK_SIZE) * BLOCK_SIZE
    return [tokens[index : index + BLOCK_SIZE] for index in range(0, total_length, BLOCK_SIZE)]


class Blocks(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, blocks: list[list[int]]) -> None:
        self._blocks = blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = torch.tensor(self._blocks[index])
        return {
            "input_ids": value,
            "attention_mask": torch.ones_like(value),
            "labels": value.clone(),
        }


tokens = [1, *([4, 5, 6, 7, 8, 9] * 10), 2, 3]
blocks = group_tokens(tokens)
print(f"source_tokens={len(tokens)} grouped_blocks={len(blocks)} block_size={BLOCK_SIZE}", flush=True)
model = BertForMaskedLM(
    BertConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=131200,
    )
)
arguments = TrainingArguments(
    output_dir=str(Path("outputs")),
    max_steps=2,
    per_device_train_batch_size=2,
    disable_tqdm=True,
    report_to=[],
    save_strategy="no",
    use_cpu=True,
)
Trainer(model=model, args=arguments, train_dataset=Blocks(blocks)).train()
print("training-complete", flush=True)
