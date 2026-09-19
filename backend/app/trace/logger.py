import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..schemas.common import TaskType
from ..schemas.trace import ExecutionTrace, ToolExecutionRecord, ValidationRecord

class TraceLogger:
    """
    Observable Execution Trace Logger.
    Strictly captures observable events: validation records, tool calls, whitelisted parameters,
    timings, computed numerical metrics, and confidence breakdowns.
    Does NOT leak internal chain-of-thought or reasoning prompts.
    """
    def __init__(self, session_id: str, task: TaskType, output_dir: Optional[str] = None):
        self.session_id = session_id
        self.task = task
        self.output_dir = Path(output_dir or "data/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        
        self.trace = ExecutionTrace(
            session_id=session_id,
            task=task,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            input_validation=ValidationRecord()
        )

    def log_validation(self, record: ValidationRecord):
        self.trace.input_validation = record

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

    def finalize(
        self,
        computed_metrics: Dict[str, Any],
        confidence_score: float,
        confidence_breakdown: Optional[Dict[str, float]] = None
    ) -> ExecutionTrace:
        total_duration = (time.time() - self.start_time) * 1000.0
        self.trace.total_duration_ms = round(total_duration, 2)
        self.trace.computed_metrics = computed_metrics
        self.trace.confidence_score = round(confidence_score, 4)
        self.trace.confidence_breakdown = confidence_breakdown or {"model": confidence_score}

        # Save trace JSON to disk
        trace_file = self.output_dir / f"{self.session_id}_trace.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(self.trace.model_dump_json(indent=2))

        return self.trace

def load_trace(session_id: str, output_dir: Optional[str] = None) -> Optional[ExecutionTrace]:
    """Retrieves an execution trace by session_id."""
    out_dir = Path(output_dir or "data/outputs")
    trace_file = out_dir / f"{session_id}_trace.json"
    if not trace_file.exists():
        return None
    with open(trace_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return ExecutionTrace(**data)
