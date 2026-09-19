from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .common import TaskType

class ValidationRecord(BaseModel):
    passed: bool = True
    crs_check: Dict[str, Any] = Field(default_factory=dict)
    overlap_iou: Optional[float] = None
    reprojections: List[Dict[str, Any]] = Field(default_factory=list)
    sar_preprocessing: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class ToolExecutionRecord(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"  # success, failed, skipped
    duration_ms: float = 0.0
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class ExecutionTrace(BaseModel):
    session_id: str
    task: TaskType
    timestamp_utc: str
    input_validation: ValidationRecord
    tool_calls: List[ToolExecutionRecord] = Field(default_factory=list)
    computed_metrics: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 0.0
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict)
    total_duration_ms: float = 0.0
