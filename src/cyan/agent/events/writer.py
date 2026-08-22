from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO

from pydantic import BaseModel

from cyan.agent.events.bus import EventBus

logger = logging.getLogger(__name__)


class EventWriter:
    # 初始化事件文件；summary_only 用于 Incident，避免重复持久化日志与 token 流
    def __init__(self, path: Path, *, summary_only: bool = False) -> None:
        self._path = path
        self._file: IO[str] | None = None
        self._summary_only = summary_only

    # 打开事件文件（追加模式），供 async with 使用
    async def __aenter__(self) -> EventWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        return self

    # 关闭事件文件
    async def __aexit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # 将事件序列化为 JSON 行并写入文件，写入失败时记录日志但不抛出异常
    async def handle(self, event: BaseModel) -> None:
        if self._file is None:
            return
        try:
            data = event.model_dump()
            if self._summary_only and data.get("type") == "llm.token":
                return
            if self._summary_only:
                event_type = data.get("type")
                mapping_field = (
                    "params"
                    if event_type in ("tool.call_started", "permission.requested")
                    else None
                )
                if mapping_field is not None:
                    value = data.pop(mapping_field, {})
                    encoded = json.dumps(value, ensure_ascii=False, default=str)
                    data["param_keys"] = (
                        sorted(value) if isinstance(value, dict) else []
                    )
                    data[f"{mapping_field}_chars"] = len(encoded)
                text_field = {
                    "run.started": "goal",
                    "tool.call_finished": "output",
                    "tool.call_failed": "error_message",
                    "log.line": "message",
                    "permission.requested": "param_preview",
                    "session.message_received": "content",
                    "skill.invoked": "arguments",
                    "subagent.started": "description",
                    "patch.proposed": "summary",
                }.get(str(event_type))
                if text_field is not None:
                    value = str(data.pop(text_field, ""))
                    data[f"{text_field}_chars"] = len(value)
                    data[f"{text_field}_bytes"] = len(value.encode("utf-8"))
            self._file.write(json.dumps(data, ensure_ascii=False) + "\n")
            self._file.flush()
        except (OSError, ValueError) as e:
            logger.error("EventWriter: failed to write event: %s", e)

    # 将 handle 注册为 bus 的订阅者
    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe(self.handle)
