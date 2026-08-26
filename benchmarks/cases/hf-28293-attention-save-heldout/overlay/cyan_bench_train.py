from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

TRAIN_STEPS = 2


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [
            {
                "input_ids": torch.tensor([1, 3 + index, 2, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "labels": torch.tensor(index % 2),
            }
            for index in range(4)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class StepCallback(TrainerCallback):
    def on_log(self, args: object, state: object, control: object, **kwargs: object) -> None:
        del args, control, kwargs
        print(f"training-step={state.global_step}", flush=True)


torch.manual_seed(43)
model = BertForSequenceClassification(
    BertConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=2,
    )
)
query = model.bert.encoder.layer[0].attention.self.query.weight
query.data = query.data.transpose(0, 1)
arguments = TrainingArguments(
    output_dir=str(Path("outputs")),
    max_steps=TRAIN_STEPS,
    per_device_train_batch_size=2,
    disable_tqdm=True,
    report_to=[],
    save_strategy="no",
    logging_steps=1,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=TinyDataset(),
    callbacks=[StepCallback()],
)
trainer.train()
print("training-main-loop-complete", flush=True)
query.data = query.data.contiguous()
trainer.save_model(str(Path("saved-model")))
print("model-save-complete", flush=True)
