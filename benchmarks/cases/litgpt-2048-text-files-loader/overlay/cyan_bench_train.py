from pathlib import Path

import torch
from litgpt.data.text_files import TextFiles


class _TinyTokenizer:
    # 将本地文本转换为固定长度的真实 token 张量
    def encode(self, text: str, *, bos: bool, eos: bool) -> torch.Tensor:
        del bos, eos
        return torch.arange(max(len(text), 32), dtype=torch.int)


# 通过 LitGPT TextFiles 与 LitData optimize 触发真实预处理路径
def main() -> None:
    root = Path("text-files-fixture")
    train = root / "train"
    validation = root / "validation"
    train.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    (train / "train.txt").write_text("training text " * 128, encoding="utf-8")
    (validation / "validation.txt").write_text("validation text " * 128, encoding="utf-8")
    data = TextFiles(train, val_data_path=validation, num_workers=0)
    data.connect(_TinyTokenizer(), batch_size=1, max_seq_length=8)
    data.prepare_data()
    next(iter(data.train_dataloader()))
    print("textfiles-preprocessing-complete", flush=True)


if __name__ == "__main__":
    main()
