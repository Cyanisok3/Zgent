from pathlib import Path

import numpy as np
from litdata import StreamingDataset, TokensLoader, optimize


def tokenize(index: int) -> np.ndarray:
    return np.arange(index, index + 64, dtype=np.int32)


def main() -> None:
    output_dir = Path("token-data")
    optimize(
        tokenize,
        inputs=list(range(128)),
        output_dir=str(output_dir),
        num_workers=1,
        chunk_bytes="64KB",
        item_loader=TokensLoader(),
    )
    print("token-preprocessing-complete", flush=True)
    dataset = StreamingDataset(
        input_dir=str(output_dir),
        item_loader=TokensLoader(block_size=16),
        shuffle=False,
    )
    first_block = dataset[0]
    print(f"token-loader-ready block-size={len(first_block)}", flush=True)


if __name__ == "__main__":
    main()
