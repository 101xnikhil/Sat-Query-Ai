import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..app.schemas.common import TaskType
from ..app.schemas.trace import ExecutionTrace, ToolExecutionRecord, ValidationRecord

class TraceLogger:
    """
    Observable Execution Trace Logger for Phase 1.
    Persists run_id, timestamp, query, task, input_validation_results, actions_taken,
    ordered tool calls with parameters/outputs/timings, confidence (null for now).
    """
    def __init__(
        self,
        run_id: Optional[str] = None,
        task: TaskType = None,
        query: str = "",
        output_dir: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        effective_id = session_id or run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.run_id = effective_id
        self.session_id = effective_id
        self.query = query
        self.task = task
        self.output_dir = Path(output_dir or "data/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        
        utc_now = datetime.now(timezone.utc).isoformat()
        self.trace = ExecutionTrace(
            session_id=effective_id,
            run_id=effective_id,
            query=query,
            task=task,
            timestamp_utc=utc_now,
            timestamp=utc_now,
            input_validation=ValidationRecord(),
            input_validation_results={},
            actions_taken=[],
            tool_calls=[],
            ordered_tool_calls=[],
            confidence=None  # Explicitly null for Phase 1
        )

    def log_validation(self, record: ValidationRecord):
        self.trace.input_validation = record
        self.trace.input_validation_results = record.model_dump()
        self.trace.actions_taken = list(record.actions_taken)

    def log_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        status: str,
        duration_ms: float,
        output_summary: Dict[str, Any],
        error: Optional[str] = None
    ):
        record = ToolExecutionRecord(
            tool_name=tool_name,
            parameters=parameters,
            status=status,
            duration_ms=round(duration_ms, 2),
            output_summary=output_summary,
            error=error
        )
        self.trace.tool_calls.append(record)
        self.trace.ordered_tool_calls.append(record)

    def finalize(
        self,
        computed_metrics: Dict[str, Any],
        confidence_score: Optional[float] = None,
        confidence_breakdown: Optional[Dict[str, float]] = None
    ) -> ExecutionTrace:
        total_duration = (time.time() - self.start_time) * 1000.0
        self.trace.total_duration_ms = round(total_duration, 2)
        self.trace.computed_metrics = computed_metrics
        # For Phase 1, confidence is null unless explicitly provided
        self.trace.confidence = confidence_score
        self.trace.confidence_score = confidence_score if confidence_score is not None else 0.0
        self.trace.confidence_breakdown = confidence_breakdown or {}

        # Save trace JSON to disk
        trace_file = self.output_dir / f"{self.run_id}_trace.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(self.trace.model_dump_json(indent=2))

        return self.trace

def load_trace(run_id: str, output_dir: Optional[str] = None) -> Optional[ExecutionTrace]:
    """Retrieves an execution trace by run_id (or session_id)."""
    out_dir = Path(output_dir or "data/outputs")
    trace_file = out_dir / f"{run_id}_trace.json"
    if not trace_file.exists():
        # Fallback to session_id pattern
        trace_file = out_dir / f"{run_id}.json"
        if not trace_file.exists():
            return None
    with open(trace_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return ExecutionTrace(**data)
