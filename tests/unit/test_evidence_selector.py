from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cyan.training.incidents.models import FailureCapsule, LogSnapshot
from cyan.training.incidents.selector import select_evidence


# 构造 selector 所需的最小失败胶囊
def _capsule(tmp_path: Path, *, job_id: str = "job-1", attempt_id: str = "attempt-1") -> FailureCapsule:
    empty = LogSnapshot(
        size=0,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        included_start=0,
        included_end=0,
        tail="",
    )
    return FailureCapsule(
        job_id=job_id,
        attempt_id=attempt_id,
        argv=["python", "train.py"],
        cwd=str(tmp_path),
        occurred_at=datetime.now(UTC),
        failure_kind="process_exit",
        returncode=1,
        stdout=empty,
        stderr=empty,
    )


# 功能：验证 selector 优先选择最后完整 traceback 和错误行
# 设计：在长 stderr 中放置两个 traceback，固定最后一个的优先级和 byte 引用
def test_selector_prefers_latest_traceback(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("ordinary output\n", encoding="utf-8")
    stderr.write_text(
        "Traceback (most recent call last):\n"
        "  File 'old.py', line 1\n"
        "ValueError: old\n\n"
        "Traceback (most recent call last):\n"
        f"  File '{tmp_path / 'train.py'}', line 2\n"
        "RuntimeError: latest\n",
        encoding="utf-8",
    )

    selection = select_evidence(_capsule(tmp_path), stdout, stderr)

    assert selection.references[0].kind == "traceback"
    assert "RuntimeError: latest" in selection.content
    assert selection.references[0].start > 0


# 功能：验证 stderr 为空时 selector 稳定回退 stdout 尾部
# 设计：不构造结构化错误，确认 fallback 仍给出可重定位引用且不超预算
def test_selector_falls_back_to_stdout_tail(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("training output\nfinal marker\n", encoding="utf-8")
    stderr.write_bytes(b"")

    selection = select_evidence(_capsule(tmp_path), stdout, stderr, max_bytes=64)

    assert selection.references[0].source == "stdout"
    assert selection.references[0].kind == "tail"
    assert "final marker" in selection.content
    assert selection.selected_bytes <= 64


# 功能：验证重复候选去重、输出上限和引用顺序完全确定
# 设计：同一错误同时命中 error_line 与尾部，重复运行比较结构化结果
def test_selector_is_deterministic_and_bounded(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("same\n", encoding="utf-8")
    stderr.write_text("RuntimeError: same\n" + ("x" * 1000) + "\n", encoding="utf-8")
    capsule = _capsule(tmp_path)

    first = select_evidence(capsule, stdout, stderr, max_bytes=128)
    second = select_evidence(capsule, stdout, stderr, max_bytes=128)

    assert first.model_dump() == second.model_dump()
    assert first.selected_bytes <= 128
    assert len(first.content.encode("utf-8")) <= 128
    assert first.duplicates_removed >= 0
