from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator
from .common import TaskType

class ValidationRecord(BaseModel):
    ok: bool = True
    passed: bool = True
    crs_check: Dict[str, Any] = Field(default_factory=dict)
    overlap_iou: Optional[float] = None
    reprojections: List[Dict[str, Any]] = Field(default_factory=list)
    sar_preprocessing: List[Dict[str, Any]] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_ok_and_passed(self) -> "ValidationRecord":
        if not self.ok or not self.passed:
            self.ok = False
            self.passed = False
        return self

class ToolExecutionRecord(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"  # success, failed, skipped
    duration_ms: float = 0.0
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class ExecutionTrace(BaseModel):
    session_id: str
    run_id: Optional[str] = None
    query: Optional[str] = None
    task: TaskType
    timestamp_utc: str
    timestamp: Optional[str] = None
    input_validation: ValidationRecord
    input_validation_results: Optional[Dict[str, Any]] = None
    actions_taken: List[str] = Field(default_factory=list)
    tool_calls: List[ToolExecutionRecord] = Field(default_factory=list)
    ordered_tool_calls: List[ToolExecutionRecord] = Field(default_factory=list)
    computed_metrics: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    confidence_score: float = 0.0
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict)
    fallback_used: bool = False
    stage_latencies: Dict[str, float] = Field(default_factory=dict)
    total_duration_ms: float = 0.0

    @model_validator(mode="after")
    def sync_aliases(self) -> "ExecutionTrace":
        if self.run_id is None:
            self.run_id = self.session_id
        if self.timestamp is None:
            self.timestamp = self.timestamp_utc
        if self.input_validation_results is None:
            self.input_validation_results = self.input_validation.model_dump()
        if not self.actions_taken and self.input_validation:
            self.actions_taken = list(self.input_validation.actions_taken)
        if not self.ordered_tool_calls and self.tool_calls:
            self.ordered_tool_calls = list(self.tool_calls)
        return self
