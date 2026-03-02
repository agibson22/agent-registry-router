from __future__ import annotations

import io
import json

import pytest

from agent_registry_router.core import RoutingEvent, StructuredLogger


def test_default_sink_is_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    logger = StructuredLogger()
    event = RoutingEvent(
        kind="classifier_run_success",
        payload={"agent": "billing", "confidence": 0.9},
    )
    logger(event)

    captured = capsys.readouterr()
    record = json.loads(captured.out.strip())
    assert record["event"] == "classifier_run_success"
    assert record["agent"] == "billing"
    assert record["confidence"] == 0.9
    assert "ts" in record


def test_logs_to_string_io() -> None:
    buf = io.StringIO()
    logger = StructuredLogger(sink=buf)
    event = RoutingEvent(
        kind="agent_run_success",
        payload={"agent": "technical"},
    )
    logger(event)

    line = buf.getvalue().strip()
    record = json.loads(line)
    assert record["event"] == "agent_run_success"
    assert record["agent"] == "technical"
    assert "ts" in record


def test_logs_error_info() -> None:
    buf = io.StringIO()
    logger = StructuredLogger(sink=buf)
    error = ValueError("something broke")
    event = RoutingEvent(
        kind="agent_resolve_failed",
        payload={"agent": "missing"},
        error=error,
    )
    logger(event)

    record = json.loads(buf.getvalue().strip())
    assert record["error"] == "something broke"
    assert record["error_type"] == "ValueError"


def test_logs_to_callable_sink() -> None:
    lines: list[str] = []
    logger = StructuredLogger(sink=lines.append)
    event = RoutingEvent(
        kind="pinned_bypass",
        payload={"agent": "billing", "message": "test"},
    )
    logger(event)

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "pinned_bypass"
    assert record["agent"] == "billing"


def test_payload_fields_merged_into_record() -> None:
    buf = io.StringIO()
    logger = StructuredLogger(sink=buf)
    event = RoutingEvent(
        kind="decision_validated",
        payload={
            "selected": "billing",
            "confidence": 0.85,
            "did_fallback": False,
        },
    )
    logger(event)

    record = json.loads(buf.getvalue().strip())
    assert record["selected"] == "billing"
    assert record["confidence"] == 0.85
    assert record["did_fallback"] is False


def test_timestamp_is_iso_format() -> None:
    buf = io.StringIO()
    logger = StructuredLogger(sink=buf)
    event = RoutingEvent(kind="test", payload={})
    logger(event)

    record = json.loads(buf.getvalue().strip())
    from datetime import datetime

    datetime.fromisoformat(record["ts"])


def test_handles_non_serializable_payload() -> None:
    buf = io.StringIO()
    logger = StructuredLogger(sink=buf)
    event = RoutingEvent(
        kind="test",
        payload={"obj": object()},
    )
    logger(event)

    record = json.loads(buf.getvalue().strip())
    assert "obj" in record
