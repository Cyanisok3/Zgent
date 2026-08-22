from __future__ import annotations

from pathlib import Path

import pytest

from cyan.agent.session.model import Session
from cyan.agent.session.store import SessionStore


# 功能：验证 SessionStore 初始化时自动创建 sessions 根目录
# 设计：传入 tmp_path 下不存在的目录，断言目录被创建，覆盖首次启动 daemon 的冷路径
def test_store_creates_root(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    SessionStore(root)
    assert root.exists()


# 功能：验证 session meta 写入后能完整读回
# 设计：构造含 run_ids 的 Session，经过 JSON 文件往返后断言字段保持，覆盖 meta.json 的持久化契约
def test_meta_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = Session(
        id="sess-1",
        mode="chat",
        status="waiting_for_input",
        title="hello",
        created_at="t1",
        updated_at="t2",
        run_ids=["run-1"],
    )
    store.write_meta(session)
    loaded = store.read_meta("sess-1")
    assert loaded == session


# 功能：验证 session meta 原子替换失败时仍保留上一份完整元数据
# 设计：先写可读基线，再令 os.replace 抛错，检查旧文件未损坏且临时文件已清理
def test_meta_replace_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path)
    original = Session(
        id="sess-1",
        mode="chat",
        status="waiting_for_input",
        title="before",
        created_at="t1",
        updated_at="t1",
    )
    store.write_meta(original)
    updated = Session(
        id="sess-1",
        mode="chat",
        status="running",
        title="after",
        created_at="t1",
        updated_at="t2",
    )

    # 模拟崩溃发生在临时文件 fsync 后、目标文件原子替换前
    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("cyan.agent.session.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        store.write_meta(updated)

    assert store.read_meta("sess-1") == original
    assert list(store.session_dir("sess-1").glob(".meta.json.*.tmp")) == []


# 功能：验证含 tool_use/tool_result block 的 thread 消息能按 Anthropic 格式读回
# 设计：追加 assistant tool_use 和 user tool_result，读取时应剥离 ts/run_id，只保留 API messages 所需字段
def test_thread_message_roundtrip_with_tool_blocks(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "read file")
    store.append_message(
        "sess-1",
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}],
        run_id="run-1",
    )
    store.append_message(
        "sess-1",
        "user",
        [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        run_id="run-1",
    )

    messages = store.read_messages("sess-1")
    assert messages == [
        {"role": "user", "content": "read file"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
    ]


# 功能：验证 thread 尾部孤儿 tool_use 会被裁掉
# 设计：构造一条未配对 tool_result 的 assistant tool_use，读取时只返回最后一次配平之前的消息，避免 API 报 messages.invalid
def test_read_messages_trims_orphan_tool_use_tail(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "hello")
    store.append_message(
        "sess-1",
        "assistant",
        [{"type": "tool_use", "id": "orphan", "name": "read_file", "input": {}}],
        run_id="run-1",
    )
    assert store.read_messages("sess-1") == [{"role": "user", "content": "hello"}]


# 功能：验证 notes.md 不存在时读为空，追加笔记后能读到内容和 run_id
# 设计：先读空状态再追加，覆盖 chat 第一轮前和 note_save 调用后的两个关键状态
def test_notes_read_and_append(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    assert store.read_notes("sess-1") == ""
    store.append_note("sess-1", "Python 3.12", "run-1")
    notes = store.read_notes("sess-1")
    assert "Python 3.12" in notes
    assert "run-1" in notes


# 功能：验证 list_meta 能恢复 Incident session 的 workspace 与关联标识
# 设计：写入两个 session 后重新枚举磁盘，断言排序无关的完整元数据集合
def test_list_meta_restores_incident_fields(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    incident = Session(
        id="sess-incident",
        mode="incident",
        status="waiting_for_input",
        title="failed training",
        created_at="t1",
        updated_at="t2",
        workspace_root="/repo",
        incident_id="inc-1",
    )
    store.write_meta(incident)

    loaded = {session.id: session for session in store.list_meta()}

    assert loaded["sess-incident"] == incident
