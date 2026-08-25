from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments


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


torch.manual_seed(11)
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
embedding = model.bert.embeddings.word_embeddings.weight
embedding.data = torch.randn(16, 16).transpose(0, 1)
arguments = TrainingArguments(
    output_dir=str(Path("outputs")),
    max_steps=2,
    per_device_train_batch_size=2,
    disable_tqdm=True,
    report_to=[],
    save_strategy="no",
    logging_steps=1,
)
trainer = Trainer(model=model, args=arguments, train_dataset=TinyDataset())
trainer.train()
print("training-main-loop-complete", flush=True)
embedding.data = embedding.data.contiguous()
trainer.save_model(str(Path("saved-model")))
print("model-save-complete", flush=True)
