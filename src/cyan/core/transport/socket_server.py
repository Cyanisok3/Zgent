from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from cyan.core.bus.envelope import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    HandlerError,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcSuccess,
    make_error,
)
from cyan.core.trace.record import TraceRecord
from cyan.core.trace.writer import TraceWriter
from cyan.core.transport.ipc_broadcaster import IpcEventBroadcaster

logger = logging.getLogger(__name__)

type CommandHandler = Callable[[dict[str, Any]], Awaitable[Any]]

# 每个连接处理协程中，当前正在处理的 writer（供 handler 读取连接上下文）
_writer_var: ContextVar[asyncio.StreamWriter] = ContextVar("_writer_var")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# 生成可安全写入 trace 的命令参数，避免持久化 job 启动环境中的密钥
def _trace_params(method: str, params: dict[str, Any]) -> dict[str, Any]:
    safe = dict(params)
    if method == "job.start" and "env" in safe:
        env = safe.get("env")
        safe["env"] = {"forwarded_keys": sorted(env) if isinstance(env, dict) else []}
    text_field = {
        "agent.run": "goal",
        "session.send_message": "content",
    }.get(method)
    if text_field is not None and text_field in safe:
        content = str(safe.pop(text_field))
        safe[f"{text_field}_chars"] = len(content)
        safe[f"{text_field}_bytes"] = len(content.encode("utf-8"))
    return safe


# 将可选映射收敛为字段名，避免 trace 复制 Incident artifact 内容
def _trace_mapping_keys(value: Any) -> list[str]:
    return sorted(value) if isinstance(value, dict) else []


# 将 Job 视图收敛为状态、字段名和计数，避免复制诊断、证据及补丁正文
def _trace_job_view(view: dict[str, Any]) -> dict[str, Any]:
    job = view.get("job")
    attempt = view.get("attempt")
    incident = view.get("incident")
    diagnosis = view.get("diagnosis")
    proposal = view.get("proposal")
    smoke_config = view.get("smoke_config")
    smoke = view.get("smoke")
    patch = view.get("patch")
    safe: dict[str, Any] = {
        "keys": sorted(view),
        "job_status": job.get("status") if isinstance(job, dict) else None,
        "job_keys": _trace_mapping_keys(job),
        "attempt_status": attempt.get("status") if isinstance(attempt, dict) else None,
        "attempt_keys": _trace_mapping_keys(attempt),
        "attempt_returncode": (
            attempt.get("returncode") if isinstance(attempt, dict) else None
        ),
        "attempt_signal": attempt.get("signal") if isinstance(attempt, dict) else None,
        "incident_status": (
            incident.get("status") if isinstance(incident, dict) else None
        ),
        "incident_keys": _trace_mapping_keys(incident),
        "diagnosis_keys": _trace_mapping_keys(diagnosis),
        "diagnosis_evidence_count": (
            len(diagnosis.get("evidence", [])) if isinstance(diagnosis, dict) else 0
        ),
        "proposal_keys": _trace_mapping_keys(proposal),
        "proposal_file_count": (
            len(proposal.get("files", [])) if isinstance(proposal, dict) else 0
        ),
        "proposal_evidence_count": (
            len(proposal.get("evidence", [])) if isinstance(proposal, dict) else 0
        ),
        "argv_count": len(view.get("argv", []))
        if isinstance(view.get("argv"), list)
        else 0,
        "smoke_config_keys": _trace_mapping_keys(smoke_config),
        "smoke_status": smoke.get("status") if isinstance(smoke, dict) else None,
        "smoke_keys": _trace_mapping_keys(smoke),
        "can_apply": bool(view.get("can_apply", False)),
    }
    if isinstance(patch, str):
        safe["patch_chars"] = len(patch)
        safe["patch_bytes"] = len(patch.encode("utf-8"))
    smoke_error = view.get("smoke_config_error")
    if isinstance(smoke_error, str):
        safe["smoke_config_error_chars"] = len(smoke_error)
        safe["smoke_config_error_bytes"] = len(smoke_error.encode("utf-8"))
    return safe


# 按 RPC 方法生成不复制日志、补丁或 session 工具输出的响应摘要
def _trace_result(method: str | None, result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if method == "job.read_log":
        data = str(result.get("data", ""))
        return {
            "data_chars": len(data),
            "data_bytes": len(data.encode("utf-8")),
            "next_offset": result.get("next_offset"),
            "total_bytes": result.get("total_bytes"),
            "eof": result.get("eof"),
        }
    if method == "job.get":
        return _trace_job_view(result)
    if method == "job.list":
        jobs = result.get("jobs")
        safe_jobs = (
            [_trace_job_view(item) for item in jobs if isinstance(item, dict)]
            if isinstance(jobs, list)
            else []
        )
        return {"jobs": safe_jobs}
    if method == "session.get_history":
        messages = result.get("messages")
        roles = (
            [str(item.get("role", "")) for item in messages if isinstance(item, dict)]
            if isinstance(messages, list)
            else []
        )
        return {"message_count": len(roles), "roles": roles}
    return result


# 生成可诊断但不含高体积或敏感 payload 的 RPC 响应 trace
def _trace_response(method: str | None, msg: BaseModel) -> dict[str, Any]:
    data = msg.model_dump()
    if method is not None:
        data["method"] = method
    if isinstance(msg, JsonRpcError):
        error = data.get("error")
        if isinstance(error, dict):
            data["error"] = {
                "code": error.get("code"),
                "message": error.get("message"),
            }
        return data
    if isinstance(msg, JsonRpcSuccess):
        data["result"] = _trace_result(method, data.get("result"))
    return data


# 返回当前 handler 调用所属连接的 StreamWriter
def get_connection_writer() -> asyncio.StreamWriter:
    return _writer_var.get()

_MAX_LINE_BYTES = 64 * 1024 * 1024  # 64 MB per frame，兼容 MCP 大文件工具结果


class SocketServer:
    def __init__(
        self,
        host: str,
        port: int,
        broadcaster: IpcEventBroadcaster | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._handlers: dict[str, CommandHandler] = {}
        self._server: asyncio.AbstractServer | None = None
        self._broadcaster = broadcaster
        self._trace = trace
        self._active_writers: set[asyncio.StreamWriter] = set()

    # 注册一个方法名对应的命令处理函数
    def register(self, method: str, handler: CommandHandler) -> None:
        self._handlers[method] = handler

    # 启动 TCP 服务器；若端口已被占用则退出进程
    async def start(self) -> str:
        try:
            _r, w = await asyncio.open_connection(self._host, self._port)
            w.close()
            await w.wait_closed()
            raise SystemExit(f"core already running at {self._host}:{self._port}")
        except (ConnectionRefusedError, OSError):
            pass

        self._server = await asyncio.start_server(
            self._accept_connection,
            host=self._host,
            port=self._port,
            limit=_MAX_LINE_BYTES,
        )
        return f"{self._host}:{self._port}"

    # 先停止接受新连接，同时保留现有 writer 以便 shutdown RPC 写回响应
    async def stop_accepting(self) -> None:
        if self._server is None:
            return
        self._server.close()
        try:
            await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass

    # 关闭服务器：先断开所有活跃连接，再等待服务器完全关闭（最多 2 秒）
    async def stop(self) -> None:
        if self._server is None:
            return
        await self.stop_accepting()
        for writer in list(self._active_writers):
            if self._broadcaster is not None:
                self._broadcaster.unsubscribe(writer)
            try:
                writer.close()
            except Exception:
                pass

    # 在调度异步连接处理前同步登记 writer，避免 shutdown 与 accept 之间的竞态
    def _accept_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> Awaitable[None]:
        self._active_writers.add(writer)
        return self._handle_connection(reader, writer)

    # 处理单个客户端连接，完成后关闭写流
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<unknown>")
        logger.debug("client connected: %s", peer)
        try:
            await self._read_loop(reader, writer)
        finally:
            self._active_writers.discard(writer)
            if self._broadcaster is not None:
                self._broadcaster.unsubscribe(writer)
            try:
                writer.close()
            except Exception:
                pass
            logger.debug("client disconnected: %s", peer)

    # 持续读取换行分隔的 JSON 行并逐行分发处理
    async def _read_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            try:
                line = await reader.readline()
            except asyncio.LimitOverrunError:
                await self._send(writer, make_error(None, INVALID_REQUEST, "Request too large"))
                return

            if not line:
                return

            # 每条命令独立作为 task 执行，避免长时间运行的 handler（如 session.send_message）
            # 阻塞读循环，使 permission.respond 等并发命令能被及时处理
            asyncio.create_task(self._handle_line(line, writer))

    # 解析单行 JSON-RPC 请求并调用对应 handler，将结果或错误写回客户端
    async def _handle_line(self, line: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as e:
            await self._send(writer, make_error(None, PARSE_ERROR, f"Parse error: {e}"))
            return

        try:
            req = JsonRpcRequest.model_validate(raw)
        except ValidationError as e:
            await self._send(writer, make_error(None, INVALID_REQUEST, "Invalid Request", str(e)))
            return

        if self._trace is not None:
            client_id = str(writer.get_extra_info("peername", "<unknown>"))
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CLIENT→CORE",
                    layer="ipc",
                    kind="command",
                    client_id=client_id,
                    data={
                        "method": req.method,
                        "id": req.id,
                        "params": _trace_params(req.method, req.params),
                    },
                )
            )

        handler = self._handlers.get(req.method)
        if handler is None:
            await self._send(
                writer,
                make_error(req.id, METHOD_NOT_FOUND, f"Method not found: {req.method}"),
                method=req.method,
            )
            return

        _writer_var.set(writer)
        try:
            result = await handler(req.params)
        except HandlerError as e:
            await self._send(
                writer,
                make_error(req.id, e.code, str(e), e.data),
                method=req.method,
            )
            return
        except ValidationError as e:
            await self._send(
                writer,
                make_error(req.id, INVALID_REQUEST, "Invalid params", str(e)),
                method=req.method,
            )
            return
        except Exception as e:
            logger.exception("handler %s raised: %s", req.method, e)
            await self._send(
                writer,
                make_error(req.id, INTERNAL_ERROR, "Internal error"),
                method=req.method,
            )
            return

        result_data: Any = result.model_dump() if isinstance(result, BaseModel) else result
        try:
            await self._send(
                writer,
                JsonRpcSuccess(id=req.id, result=result_data),
                method=req.method,
            )
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("client disconnected before response for %s", req.method)

    # 将 pydantic 消息序列化为 JSON 行并写入流，随后刷新缓冲区
    async def _send(
        self,
        writer: asyncio.StreamWriter,
        msg: BaseModel,
        *,
        method: str | None = None,
    ) -> None:
        writer.write(msg.model_dump_json().encode() + b"\n")
        await writer.drain()
        if self._trace is not None:
            kind = "error" if isinstance(msg, JsonRpcError) else "response"
            client_id = str(writer.get_extra_info("peername", "<unknown>"))
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CORE→CLIENT",
                    layer="ipc",
                    kind=kind,
                    client_id=client_id,
                    data=_trace_response(method, msg),
                )
            )
