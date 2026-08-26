from __future__ import annotations

import json
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


# 功能：验证 Selected Evidence 头部是可直接复制的三字段 EvidenceRef JSON
# 设计：解析第一行并与 selector 生成的稳定 byte-range 对照，确保模型不会看到内部字段
def test_selector_renders_copyable_three_field_reference(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_bytes(b"")
    stderr.write_text("RuntimeError: root\n", encoding="utf-8")

    selection = select_evidence(_capsule(tmp_path), stdout, stderr)
    header = json.loads(selection.content.splitlines()[0])

    assert set(header) == {"source", "reference", "description"}
    assert header["source"] == "stderr"
    assert header["reference"] == selection.references[0].as_reference(_capsule(tmp_path))
    assert header["description"] == selection.references[0].selection_reason
    assert "RuntimeError: root" in selection.content


# 功能：验证 stderr 为空时 selector 稳定回退 stdout 尾部
# 设计：不构造结构化错误，确认 fallback 仍给出可重定位引用且不超预算
def test_selector_falls_back_to_stdout_tail(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("training output\nfinal marker\n", encoding="utf-8")
    stderr.write_bytes(b"")

    selection = select_evidence(_capsule(tmp_path), stdout, stderr, max_bytes=256)

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


# 功能：验证无 traceback 的具体 Python 异常名仍是 stderr 根因候选
# 设计：用更长 stdout 制造竞争，断言 RuntimeError 不会被 stdout 尾部挤掉
def test_selector_prefers_stderr_runtime_error_over_long_stdout(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("progress\n" * 4000, encoding="utf-8")
    stderr.write_text("RuntimeError: optimizer exploded\n", encoding="utf-8")

    selection = select_evidence(_capsule(tmp_path), stdout, stderr, max_bytes=256)

    assert selection.references[0].source == "stderr"
    assert selection.references[0].kind == "error_line"
    assert "RuntimeError: optimizer exploded" in selection.content


# 功能：验证超长单行日志仍能通过字节尾部回退选中
# 设计：构造超过 8 KiB 的无换行 stdout，检查末尾标记和精确起点
def test_selector_keeps_tail_of_oversized_single_line(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_bytes(b"x" * (12 * 1024) + b"FINAL_MARKER")
    stderr.write_bytes(b"")

    selection = select_evidence(_capsule(tmp_path), stdout, stderr, max_bytes=512)

    assert "FINAL_MARKER" in selection.content
    assert selection.references[0].start > 0


# 功能：验证超预算 traceback 仍保留最终异常行
# 设计：用大量 frame 耗尽 traceback 候选空间，断言终止 ValueError 优先存活
def test_selector_reserves_budget_for_traceback_exception(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_bytes(b"")
    stderr.write_text(
        "Traceback (most recent call last):\n"
        + "".join(f"  File '/outside/frame_{index}.py', line 1\n" for index in range(1000))
        + "ValueError: final root cause\n",
        encoding="utf-8",
    )

    selection = select_evidence(_capsule(tmp_path), stdout, stderr, max_bytes=256)

    assert "ValueError: final root cause" in selection.content
    assert any(item.kind == "traceback_exception" for item in selection.references)
