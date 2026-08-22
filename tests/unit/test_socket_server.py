from __future__ import annotations

import asyncio
import json
import socket

from cyan.service.protocol.envelope import JsonRpcSuccess, make_error
from cyan.service.transport.socket_server import SocketServer, _trace_params, _trace_response


# 分配一个当前可用的本地 TCP 端口
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# 功能：验证客户端断开后 SocketServer 调用 broadcaster.unsubscribe(writer) 清理订阅
# 设计：用内联 MockBroadcaster 捕获 unsubscribe 调用并设置 asyncio.Event，避免 sleep 轮询；
#       等待 Event 而非断言调用次数，确保时序正确性而不依赖竞态假设
async def test_broadcaster_unsubscribe_called_on_disconnect() -> None:
    unsubscribed = asyncio.Event()

    class MockBroadcaster:
        # 记录连接清理时的退订调用
        def unsubscribe(self, writer: object) -> None:
            unsubscribed.set()

    port = _free_port()
    server = SocketServer("127.0.0.1", port, broadcaster=MockBroadcaster())  # type: ignore[arg-type]
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()
        await writer.wait_closed()

        await asyncio.wait_for(unsubscribed.wait(), timeout=2.0)
    finally:
        await server.stop()


# 功能：验证不传入 broadcaster 时 SocketServer 仍可正常启动和停止（backward-compatible 默认值）
# 设计：直接实例化 SocketServer(host, port)（无 broadcaster），start/stop 不抛异常即为通过；
#       回归测试确保新参数的默认值 None 不破坏现有调用方
async def test_no_broadcaster_server_starts_and_stops() -> None:
    port = _free_port()
    server = SocketServer("127.0.0.1", port)
    await server.start()
    await server.stop()


# 功能：验证服务器主动停止时会在关闭连接前取消该连接的事件订阅
# 设计：先完成一次协议往返，确定连接已被 server 接管，再调用 stop 观察退订事件
async def test_server_stop_unsubscribes_active_connections() -> None:
    unsubscribed = asyncio.Event()

    class MockBroadcaster:
        # 记录服务器停止路径触发的退订调用
        def unsubscribe(self, writer: object) -> None:
            unsubscribed.set()

    port = _free_port()
    server = SocketServer("127.0.0.1", port, broadcaster=MockBroadcaster())  # type: ignore[arg-type]
    await server.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"not-json\n")
    await writer.drain()
    await asyncio.wait_for(reader.readline(), timeout=1.0)

    try:
        await server.stop()
        await asyncio.wait_for(unsubscribed.wait(), timeout=1.0)
    finally:
        writer.close()
        await writer.wait_closed()


# 功能：验证 job.read_log 响应 trace 只保留 offset、EOF 与体积摘要
# 设计：在真实 JSON-RPC 成功模型中放入唯一日志标记，确认 wire 内容不变而 trace 摘要无原文
def test_job_log_response_trace_redacts_raw_data() -> None:
    secret = "RAW-TRAINING-LOG-秘密"
    response = JsonRpcSuccess(
        id="rpc-1",
        result={
            "data": secret,
            "next_offset": 42,
            "total_bytes": 128,
            "eof": False,
        },
    )

    traced = _trace_response("job.read_log", response)

    assert secret not in json.dumps(traced, ensure_ascii=False)
    assert traced["method"] == "job.read_log"
    assert traced["result"] == {
        "data_chars": len(secret),
        "data_bytes": len(secret.encode()),
        "next_offset": 42,
        "total_bytes": 128,
        "eof": False,
    }
    assert response.result["data"] == secret


# 功能：验证 job 视图、session 历史与错误 trace 均不会复制 Incident 或工具输出
# 设计：复用同一敏感标记覆盖诊断、补丁、history 和错误 data 四条响应路径
def test_sensitive_rpc_response_trace_uses_summaries() -> None:
    secret = "SECRET-PATCH-OR-TOOL-OUTPUT"
    job_view = {
        "job": {"status": "failed", "attempt_ids": ["attempt-1"]},
        "attempt": {"status": "failed", "returncode": 7, "signal": None},
        "incident": {"status": "awaiting_approval", "failure_path": secret},
        "diagnosis": {
            "summary": secret,
            "root_cause": secret,
            "evidence": [{"description": secret}],
        },
        "proposal": {
            "files": [{"path": secret}],
            "evidence": [{"description": secret}],
        },
        "argv": ["python", secret],
        "workspace_root": secret,
        "patch": secret,
        "smoke_config": {"argv": [secret], "timeout_s": 5},
        "smoke": {"status": "failed", "stderr": secret},
        "can_apply": True,
        "smoke_config_error": secret,
    }
    responses = [
        _trace_response("job.get", JsonRpcSuccess(id="1", result=job_view)),
        _trace_response("job.list", JsonRpcSuccess(id="2", result={"jobs": [job_view]})),
        _trace_response(
            "session.get_history",
            JsonRpcSuccess(
                id="3",
                result={"messages": [{"role": "user", "content": secret}]},
            ),
        ),
        _trace_response("job.start", make_error("4", -32602, "bad input", secret)),
    ]

    assert secret not in json.dumps(responses)
    assert responses[0]["result"]["patch_chars"] == len(secret)
    assert responses[1]["result"]["jobs"][0]["patch_bytes"] == len(secret.encode())
    assert responses[0]["result"]["job_status"] == "failed"
    assert responses[0]["result"]["incident_status"] == "awaiting_approval"
    assert responses[0]["result"]["diagnosis_evidence_count"] == 1
    assert responses[0]["result"]["proposal_file_count"] == 1
    assert responses[0]["result"]["attempt_returncode"] == 7
    assert responses[0]["result"]["argv_count"] == 2
    assert responses[0]["result"]["smoke_config_error_chars"] == len(secret)
    assert responses[2]["result"] == {"message_count": 1, "roles": ["user"]}
    assert "data" not in responses[3]["error"]


# 功能：验证用户 goal、session 消息和启动环境进入命令 trace 前不会保留原值
# 设计：分别调用纯摘要函数并序列化检查唯一敏感标记，同时保留长度和环境键名用于诊断
def test_sensitive_rpc_command_trace_redacts_content_and_env_values() -> None:
    secret = "USER-OR-ENV-SECRET-秘密"
    traced = [
        _trace_params("agent.run", {"goal": secret}),
        _trace_params(
            "session.send_message",
            {"session_id": "session-1", "content": secret},
        ),
        _trace_params(
            "job.start",
            {"argv": ["python"], "env": {"API_TOKEN": secret}},
        ),
        _trace_params(
            "launch.preview",
            {"command": secret, "env": {"API_TOKEN": secret}},
        ),
        _trace_params(
            "launch.start",
            {
                "command": secret,
                "env": {"API_TOKEN": secret},
                "preview_fingerprint": "a" * 64,
            },
        ),
    ]

    assert secret not in json.dumps(traced, ensure_ascii=False)
    assert traced[0]["goal_chars"] == len(secret)
    assert traced[1]["content_bytes"] == len(secret.encode())
    assert traced[2]["env"] == {"forwarded_keys": ["API_TOKEN"]}
    assert traced[3]["command_chars"] == len(secret)
    assert traced[3]["env"] == {"forwarded_keys": ["API_TOKEN"]}
    assert traced[4]["command_bytes"] == len(secret.encode())


# 功能：验证 incident.review 响应 trace 不复制审阅前后的源码
# 设计：使用唯一敏感标记构造成功响应并检查仅保留路径与体积摘要
def test_incident_review_trace_redacts_source_text() -> None:
    secret = "PRIVATE-SOURCE-TEXT"
    response = JsonRpcSuccess(
        id="review-1",
        result={
            "proposal_id": "proposal-1",
            "path": "train.py",
            "before_text": secret,
            "after_text": f"{secret}-updated",
        },
    )

    traced = _trace_response("incident.review", response)

    assert secret not in json.dumps(traced)
    assert traced["result"]["proposal_id"] == "proposal-1"
    assert traced["result"]["before_chars"] == len(secret)
