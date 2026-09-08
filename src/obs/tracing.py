"""轻量观测日志:trace_id 贯穿 + 各阶段延迟 / token / 费用 / chunk_id / 异常。

约束 7 要求:每次查询可追溯 —— trace_id、各阶段延迟、token、估算费用、
召回 chunk_id、最终引用 chunk_id、异常信息。
实现原则:结构化 JSON 日志(logging.info),不落文件、不影响业务;可在
config.yaml 用 logging.trace 开关控制。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("docpilot.trace")


class Tracer:
    """一次查询的观测上下文。"""

    def __init__(self, enabled: bool = True) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.enabled = enabled
        self._steps: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def step(self, name: str, **fields: Any) -> None:
        """记录一个阶段:名称 + 相对起始耗时 + 附加字段。"""
        self._steps.append(
            {"step": name, "elapsed_s": round(time.perf_counter() - self._t0, 4), **fields}
        )

    def emit(self, **extra: Any) -> None:
        """输出一次完整查询记录(JSON 行)。"""
        if not self.enabled:
            return
        record = {"trace_id": self.trace_id, "steps": self._steps, **extra}
        logger.info(json.dumps(record, ensure_ascii=False, default=str))

    def summary(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "steps": self._steps}
