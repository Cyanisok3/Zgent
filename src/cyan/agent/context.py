from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionContext:
    run_id: str
    goal: str
    max_steps: int
    system_prompt_text: str = ""
    max_input_bytes: int = 128 * 1024
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    status: str = "running"
    reason: str | None = None
    result: str = ""
    initial_input_bytes: int = 0
    peak_input_bytes: int = 0
    budget_exhausted: bool = False

    # 初始化本轮唯一的用户指令，禁止隐式历史会话注入
    def __post_init__(self) -> None:
        if not self.messages:
            self.messages.append({"role": "user", "content": self.goal})

    # 返回固定 Incident system prompt，不拼接全局记忆或会话笔记
    def system_prompt(self, _base: str = "") -> str:
        return self.system_prompt_text

    # 将 LLM 响应内容追加到当前运行消息
    def add_assistant_message(self, content: list[Any]) -> None:
        self.messages.append({"role": "assistant", "content": content})

    # 将工具结果加入上下文，超限时仅保留结构化错误占位
    def add_tool_result(
        self,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
        *,
        tool_schemas: list[dict[str, object]] | None = None,
    ) -> None:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        candidate = [*self.messages]
        last = candidate[-1] if candidate else None
        if (
            last is not None
            and last.get("role") == "user"
            and isinstance(last.get("content"), list)
            and last.get("content")
            and all(
                isinstance(item, dict) and item.get("type") == "tool_result"
                for item in last["content"]
            )
        ):
            last["content"] = [*last["content"], block]
        else:
            candidate.append({"role": "user", "content": [block]})
        if self.serialized_size(tool_schemas or [], messages=candidate) > self.max_input_bytes:
            self.budget_exhausted = True
            block["content"] = json.dumps(
                {
                    "error": "context_budget_exhausted",
                    "tool_use_id": tool_use_id,
                },
                separators=(",", ":"),
            )
            block["is_error"] = True
            if last is None or last.get("role") != "user":
                candidate[-1] = {"role": "user", "content": [block]}
        self.messages = candidate

    # 计算 system、messages 和 tools 序列化后的 UTF-8 总大小
    def serialized_size(
        self,
        tool_schemas: list[dict[str, object]],
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> int:
        payload = {
            "system": self.system_prompt_text,
            "messages": self.messages if messages is None else messages,
            "tools": tool_schemas,
        }
        return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))

    # 记录一次发送前输入大小，供 Incident run 评测使用
    def observe_input_size(self, size: int) -> None:
        if self.initial_input_bytes == 0:
            self.initial_input_bytes = size
        self.peak_input_bytes = max(self.peak_input_bytes, size)

    # 返回 True 表示当前运行已经结束
    def is_done(self) -> bool:
        return self.status != "running"

    # 将运行标记为成功并保留模型最终文字
    def mark_success(self) -> None:
        self.status = "success"

    # 将运行标记为失败并记录可审计原因
    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.reason = reason
