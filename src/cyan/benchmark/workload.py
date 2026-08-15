from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Literal

Domain = Literal["data_processing", "feature_transform", "cpu_training"]
Failure = Literal[
    "main_nonzero",
    "check_failed",
    "postflight_violation",
    "deterministic_input",
]


# 执行四种真实 Pandas 数据变换之一
def _data_operation(adapter: int, *, broken_main: bool) -> int:
    import pandas as pd  # type: ignore[import-untyped]

    if adapter == 0:
        left = pd.DataFrame({"user_id": [1, 2, 3], "value": [4.0, 5.0, 6.0]})
        right_ids = ["1", "2", "3"] if broken_main else [1, 2, 3]
        right = pd.DataFrame({"user_id": right_ids, "label": [0, 1, 0]})
        return len(left.merge(right, on="user_id", validate="one_to_one"))
    if adapter == 1:
        values = ["2026-01-01", "not-a-date"] if broken_main else ["2026-01-01", "2026-01-02"]
        frame = pd.DataFrame({"timestamp": values, "value": [1.0, 2.0]})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        return len(frame.set_index("timestamp").resample("1D").sum())
    if adapter == 2:
        frame = pd.DataFrame(
            {
                "sample": ["a", "a" if broken_main else "b"],
                "feature": ["x", "x"],
                "value": [1.0, 2.0],
            }
        )
        return int(frame.pivot(index="sample", columns="feature", values="value").size)
    frame = pd.DataFrame({"group": ["a", "a", "b"], "value": [1.0, 2.0, 3.0]})
    window: int | str = "invalid" if broken_main else 2
    return int(frame.groupby("group")["value"].rolling(window).mean().notna().sum())


# 执行四种真实 scikit-learn 特征变换之一
def _feature_operation(adapter: int, *, broken_main: bool) -> int:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

    if adapter == 0:
        raw: list[list[float | str]] = [[1.0, 2.0], [3.0, "bad" if broken_main else 4.0]]
        return int(StandardScaler().fit_transform(raw).size)
    if adapter == 1:
        encoder = OneHotEncoder(handle_unknown="error" if broken_main else "ignore")
        encoder.fit([["red"], ["blue"]])
        return int(encoder.transform([["green"]]).shape[1])
    if adapter == 2:
        train = pd.DataFrame({"age": [20, 30], "income": [1.0, 2.0]})
        transformer = ColumnTransformer([("numeric", StandardScaler(), ["age", "income"])])
        transformer.fit(train)
        test = train.drop(columns=["income"]) if broken_main else train
        return int(transformer.transform(test).size)
    documents = (
        ["the and", "and the"]
        if broken_main
        else ["cyan finds causal evidence", "logs show failure"]
    )
    pipeline = Pipeline([("tfidf", TfidfVectorizer(stop_words="english"))])
    return int(np.asarray(pipeline.fit_transform(documents).sum()).size)


# 执行四种真实 PyTorch CPU 计算之一
def _training_operation(adapter: int, *, broken_main: bool, output: Path) -> int:
    import torch

    torch.manual_seed(7)
    if adapter == 0:
        features = torch.randn(16, 3 if broken_main else 4)
        model = torch.nn.Linear(4, 2)
        return int(model(features).numel())
    if adapter == 1:
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 4 if broken_main else 1])
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()  # type: ignore[no-untyped-call]
        return int(logits.numel())
    if adapter == 2:
        sequence = torch.randn(2, 5, 3 if broken_main else 4)
        network = torch.nn.LSTM(input_size=4, hidden_size=6, batch_first=True)
        transformed, _state = network(sequence)
        return int(transformed.numel())
    source = torch.nn.Linear(4, 2)
    checkpoint = output.with_name("checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source.state_dict(), checkpoint)
    target = torch.nn.Linear(5 if broken_main else 4, 2)
    target.load_state_dict(torch.load(checkpoint, weights_only=True))
    return sum(parameter.numel() for parameter in target.parameters())


# 执行一个真实 Data/ML adapter 并返回可检查的结果规模
def _operation(domain: Domain, adapter: int, *, broken_main: bool, output: Path) -> int:
    if domain == "data_processing":
        return _data_operation(adapter, broken_main=broken_main)
    if domain == "feature_transform":
        return _feature_operation(adapter, broken_main=broken_main)
    return _training_operation(adapter, broken_main=broken_main, output=output)


# 在实际失败之后追加确定性清理日志，使根因离开 Capsule 尾部
def _cleanup_noise(domain: Domain, adapter: int) -> None:
    for index in range(1800):
        print(f"cleanup domain={domain} adapter={adapter} task={index}", file=sys.stderr)


# 执行 preflight、真实主操作和 postflight 检查
def run_workload(
    domain: Domain,
    adapter: int,
    failure: Failure,
    *,
    fixed: bool,
    workspace: Path,
) -> int:
    input_path = workspace / "data" / "input.csv"
    output_path = workspace / "artifacts" / "result.bin"
    if fixed:
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("feature,label\n1,0\n", encoding="utf-8")
    try:
        if failure == "deterministic_input" and not input_path.is_file():
            raise FileNotFoundError(f"required workflow input is missing: {input_path.name}")
        size = _operation(
            domain,
            adapter,
            broken_main=failure == "main_nonzero" and not fixed,
            output=output_path,
        )
        print(json.dumps({"domain": domain, "adapter": adapter, "result_size": size}))
        if failure == "check_failed":
            required = size + 1 if not fixed else max(1, size)
            if size < required:
                raise AssertionError(
                    f"quality check failed: result_size={size} required_min={required}"
                )
        if failure == "postflight_violation":
            if fixed:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"verified-workflow-artifact")
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("postflight artifact contract failed: result.bin is missing")
    except Exception:
        traceback.print_exc()
        _cleanup_noise(domain, adapter)
        return 2
    print("workflow verification completed")
    return 0


# 解析参数并运行独立的真实 Benchmark workload
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=["data_processing", "feature_transform", "cpu_training"],
        required=True,
    )
    parser.add_argument("--adapter", type=int, choices=range(4), required=True)
    parser.add_argument(
        "--failure",
        choices=["main_nonzero", "check_failed", "postflight_violation", "deterministic_input"],
        required=True,
    )
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_workload(
            args.domain,
            args.adapter,
            args.failure,
            fixed=args.fixed,
            workspace=Path.cwd(),
        )
    )


if __name__ == "__main__":
    main()
