import pytest
from backend.app.schemas.common import Modality, TaskType, ImageMetadata
from backend.app.schemas.query import QueryRequest
from backend.app.controller.llm_planner import LLMPlanner
from backend.app.controller.engine import ControllerEngine

def test_multistep_planner_generates_dependency_chain():
    """Verify multi-step query generates change_map -> optical_sar_fusion -> change_vqa dependency chain."""
    planner = LLMPlanner()
    opt_meta = ImageMetadata(
        image_id="opt_scene",
        filename="opt_scene.tif",
        file_path="data/opt_scene.tif",
        format="geotiff",
        width=512,
        height=512,
        crs="EPSG:4326",
        band_count=3,
        dtype="uint8",
        resolution=(10.0, 10.0),
        modality=Modality.OPTICAL,
        modality_confidence=0.98
    )
    sar_meta = ImageMetadata(
        image_id="sar_scene",
        filename="sar_scene.tif",
        file_path="data/sar_scene.tif",
        format="geotiff",
        width=512,
        height=512,
        crs="EPSG:4326",
        band_count=1,
        dtype="float32",
        resolution=(10.0, 10.0),
        modality=Modality.SAR,
        modality_confidence=0.95
    )
    
    task, plan, fallback_used, actions = planner.plan(
        query="what changed and is the new area water or built-up?",
        images=[opt_meta, sar_meta]
    )
    assert task == TaskType.CHANGE_VQA
    assert len(plan) == 3
    tool_names = [p[0] for p in plan]
    assert tool_names == ["change_map", "optical_sar_fusion", "change_vqa"]

def test_multistep_compound_execution_end_to_end(optical_geotiff, sar_geotiff):
    """End-to-end engine execution for compound change + fusion query."""
    engine = ControllerEngine()
    
    req = QueryRequest(
        query="what changed and is the new area water or built-up?",
        image_ids=[optical_geotiff, sar_geotiff]
    )
    resp = engine.process_query(req)
    assert resp.task == TaskType.CHANGE_VQA
    assert resp.confidence_score > 0.0
    assert resp.trace.stage_latencies.get("planning_ms", 0.0) >= 0.0
    assert resp.trace.stage_latencies.get("tool_execution_ms", 0.0) >= 0.0
    assert resp.trace.stage_latencies.get("synthesis_ms", 0.0) >= 0.0
    assert "confidence_breakdown" in resp.trace.model_dump()
    assert "token_probability" in resp.trace.confidence_breakdown
    assert "input_quality" in resp.trace.confidence_breakdown
