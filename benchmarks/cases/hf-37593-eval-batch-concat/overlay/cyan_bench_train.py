from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments


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


def compute_metrics(eval_prediction: object) -> dict[str, float]:
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    predicted_labels = np.argmax(predictions, axis=1)
    return {"accuracy": float(np.mean(predicted_labels == labels))}


torch.manual_seed(7)
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
    max_steps=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_do_concat_batches=False,
    disable_tqdm=True,
    report_to=[],
    save_strategy="no",
    logging_steps=1,
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=TinyDataset(4),
    eval_dataset=TinyDataset(5),
    compute_metrics=compute_metrics,
)
trainer.train()
print("training-main-loop-complete", flush=True)
metrics = trainer.evaluate()
print(f"evaluation-complete accuracy={metrics['eval_accuracy']}", flush=True)
