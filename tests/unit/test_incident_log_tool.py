from __future__ import annotations

import json

from cyan.training.incidents.log_tool import LogStream, ReadJobLogTool


class MemoryLogReader:
    # 使用内存字节串初始化可控日志 reader
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bytes_read = 0

    # 返回内存日志长度
    async def size(self, job_id: str, attempt_id: str, stream: LogStream) -> int:
        return len(self.data)

    # 按稳定字节偏移返回内存日志片段
    async def read(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int,
        limit: int,
    ) -> bytes:
        content = self.data[offset : offset + limit]
        self.bytes_read += len(content)
        return content

    # 从指定偏移返回内存日志中的首个查询串位置
    async def search(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int,
        query: bytes,
    ) -> int | None:
        found = self.data.find(query, offset)
        return found if found >= 0 else None


# 功能：验证 tail 模式最多返回 32 KiB 且标注精确字节范围
# 设计：使用 40 KiB 单字节日志避开字符编码干扰，直接断言 start、end 和返回内容长度
async def test_read_job_log_tail_is_bounded() -> None:
    tool = ReadJobLogTool(MemoryLogReader(b"x" * (40 * 1024)))

    result = await tool.invoke(
        {
            "job_id": "job-1",
            "attempt_id": "attempt-1",
            "stream": "stderr",
            "mode": "tail",
        }
    )

    payload = json.loads(result.content)
    assert payload["slice"]["start"] == 8 * 1024
    assert payload["slice"]["end"] == 40 * 1024
    assert len(payload["slice"]["content"].encode()) == 32 * 1024


# 功能：验证 range 模式使用调用者给出的稳定 byte offset
# 设计：读取中间四个字节并断言闭开区间，避免把字符行号误当成日志证据定位
async def test_read_job_log_range_uses_byte_offsets() -> None:
    tool = ReadJobLogTool(MemoryLogReader(b"0123456789"))

    result = await tool.invoke(
        {
            "job_id": "job-1",
            "attempt_id": "attempt-1",
            "stream": "stdout",
            "mode": "range",
            "offset": 3,
            "limit": 4,
        }
    )

    payload = json.loads(result.content)
    assert payload["slice"] == {"start": 3, "end": 7, "content": "3456"}


# 功能：验证 search 能发现跨内部扫描块边界的错误文本
# 设计：把查询串刻意放在 64 KiB 边界两侧，覆盖 overlap 算法而不暴露 reader 实现
async def test_read_job_log_search_crosses_chunk_boundary() -> None:
    prefix = b"x" * (64 * 1024 - 3)
    tool = ReadJobLogTool(MemoryLogReader(prefix + b"ERROR" + b"z" * 100))

    result = await tool.invoke(
        {
            "job_id": "job-1",
            "attempt_id": "attempt-1",
            "stream": "stderr",
            "mode": "search",
            "query": "ERROR",
            "limit": 64,
        }
    )

    payload = json.loads(result.content)
    assert payload["match_offset"] == len(prefix)
    assert "ERROR" in payload["slice"]["content"]
    assert payload["slice"]["end"] - payload["slice"]["start"] <= 64


# 功能：验证全文搜索只把返回片段计入证据读取量
# 设计：把匹配放到 256 KiB 之后，并用 reader 计数确认扫描内容未经过受预算 read
async def test_read_job_log_search_only_reads_returned_evidence() -> None:
    marker_offset = 512 * 1024
    reader = MemoryLogReader(b"x" * marker_offset + b"ROOT_CAUSE" + b"z" * 1024)
    tool = ReadJobLogTool(reader)

    result = await tool.invoke(
        {
            "job_id": "job-1",
            "attempt_id": "attempt-1",
            "stream": "stderr",
            "mode": "search",
            "query": "ROOT_CAUSE",
            "limit": 128,
        }
    )

    payload = json.loads(result.content)
    assert payload["match_offset"] == marker_offset
    assert "ROOT_CAUSE" in payload["slice"]["content"]
    assert reader.bytes_read == 128


# 功能：验证 search 未提供 query 时返回 schema_error
# 设计：直接调用工具覆盖独立业务约束，确保没有空查询导致全日志扫描
async def test_read_job_log_search_requires_query() -> None:
    tool = ReadJobLogTool(MemoryLogReader(b"content"))

    result = await tool.invoke(
        {
            "job_id": "job-1",
            "attempt_id": "attempt-1",
            "stream": "stderr",
            "mode": "search",
        }
    )

    assert result.is_error
    assert result.error_type == "schema_error"
