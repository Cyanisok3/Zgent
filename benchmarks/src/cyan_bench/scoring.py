from __future__ import annotations

import json
from pathlib import Path

from cyan_bench.models import ResolvedAnchor, SelectionArtifact, SelectionReference


# 判断一个选择引用是否覆盖任意 gold evidence 字节
def _overlaps(reference: SelectionReference, gold: ResolvedAnchor) -> bool:
    return (
        reference.source == gold.source
        and reference.start < gold.end
        and gold.start < reference.end
    )


# 计算一个 selection 对全部 required evidence group 的覆盖指标
def score_selection(selection: SelectionArtifact, gold_path: Path) -> dict[str, object]:
    gold = [
        ResolvedAnchor.model_validate(item)
        for item in json.loads(gold_path.read_text(encoding="utf-8"))
    ]
    group_ids = sorted({item.group for item in gold})
    hit_groups = {
        group
        for group in group_ids
        if any(
            _overlaps(reference, item)
            for reference in selection.references
            for item in gold
            if item.group == group
        )
    }
    first_hit = next(
        (
            index
            for index, reference in enumerate(selection.references, 1)
            if any(_overlaps(reference, item) for item in gold)
        ),
        None,
    )
    total = max(1, selection.scanned_bytes)
    recall = len(hit_groups) / len(group_ids) if group_ids else 1.0
    return {
        "case_id": selection.case_id,
        "baseline": selection.baseline,
        "repeat": selection.repeat,
        "required_group_count": len(group_ids),
        "required_groups_hit": len(hit_groups),
        "required_group_recall": round(recall, 6),
        "gold_evidence_hit": recall == 1.0,
        "first_hit_rank": first_hit,
        "selection_ratio": round(selection.unique_selected_bytes / total, 6),
        "reduction_ratio": round(1 - selection.unique_selected_bytes / total, 6),
        "selector_latency_seconds": selection.latency_seconds,
    }
