from __future__ import annotations

from typing import Any


class HandlerError(Exception):
    """应用处理器可向服务层传递的结构化错误。"""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data
