import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..schemas.query import QueryRequest, QueryResponse
from ..controller.engine import ControllerEngine
from ..config import get_settings

router = APIRouter(prefix="/batch", tags=["batch"])
engine = ControllerEngine()

class BatchEvaluationRequest(BaseModel):
    items: List[QueryRequest]

class BatchEvaluationResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]

@router.post("", response_model=BatchEvaluationResponse)
@router.post("/evaluate", response_model=BatchEvaluationResponse)
def run_batch_evaluation(batch_req: BatchEvaluationRequest):
    results = []
    success_count = 0
    fail_count = 0

    for req in batch_req.items:
        try:
            res: QueryResponse = engine.process_query(req)
            results.append({
                "session_id": res.session_id,
                "query": res.query,
                "task": res.task.value,
                "status": "success",
                "answer": res.answer,
                "confidence": res.confidence_score,
                "computed_metrics": res.computed_metrics,
                "total_duration_ms": res.trace.total_duration_ms
            })
            success_count += 1
        except Exception as e:
            results.append({
                "session_id": req.session_id or "unknown",
                "query": req.query,
                "status": "failed",
                "error": str(e)
            })
            fail_count += 1

    return BatchEvaluationResponse(
        total=len(batch_req.items),
        successful=success_count,
        failed=fail_count,
        results=results
    )
