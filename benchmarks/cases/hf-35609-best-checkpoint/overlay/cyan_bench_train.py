from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class TinyDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.samples = [
            {
                "input_ids": torch.tensor([1, 3 + index % 8, 2, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "labels": torch.tensor(index % 2),
            }
            for index in range(size)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class StepCallback(TrainerCallback):
    def on_step_end(self, args: object, state: object, control: object, **kwargs: object) -> None:
        del args, control, kwargs
        print(f"training-step={state.global_step}", flush=True)


metric_values = iter((1.0, 2.0, 0.0))


def compute_metrics(eval_prediction: object) -> dict[str, float]:
    predictions = np.argmax(eval_prediction.predictions, axis=1)
    labels = eval_prediction.label_ids
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "score": next(metric_values),
    }


torch.manual_seed(29)
model = BertForSequenceClassification(
    BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=2,
    )
)
arguments = TrainingArguments(
    output_dir=str(Path("outputs")),
    max_steps=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=1,
    save_total_limit=1,
    metric_for_best_model="score",
    greater_is_better=True,
    disable_tqdm=True,
    report_to=[],
    logging_steps=1,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=TinyDataset(6),
    eval_dataset=TinyDataset(4),
    compute_metrics=compute_metrics,
    callbacks=[StepCallback()],
)
trainer.train()
print("training-and-checkpointing-complete", flush=True)
