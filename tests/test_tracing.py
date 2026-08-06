"""观测日志测试:阶段记录、字段完整、开关控制。"""

from __future__ import annotations

import logging

from obs import Tracer


def test_tracer_records_steps(caplog) -> None:
    tracer = Tracer()
    tracer.step("query_prep", translated_query="synthetic CT")
    tracer.step("retrieve", recall_chunk_ids=["a_c0001", "b_c0001"])
    tracer.step("generate", cited_chunk_ids=["a_c0001"], cost_yuan=0.001)
    summary = tracer.summary()
    assert summary["trace_id"]
    assert [s["step"] for s in summary["steps"]] == ["query_prep", "retrieve", "generate"]
    assert summary["steps"][0]["translated_query"] == "synthetic CT"
    # 时间戳递增
    times = [s["elapsed_s"] for s in summary["steps"]]
    assert times == sorted(times)


def test_tracer_emit_writes_json_line(caplog) -> None:
    tracer = Tracer()
    tracer.step("retrieve", recall_chunk_ids=["x"])
    with caplog.at_level(logging.INFO, logger="medidoc.trace"):
        tracer.emit(error=None)
    assert any("trace_id" in r.message for r in caplog.records)


def test_tracer_disabled() -> None:
    tracer = Tracer(enabled=False)
    tracer.step("retrieve")
    assert tracer.summary()["steps"]  # 内部仍记录(便于测试);emit 才受开关控制
