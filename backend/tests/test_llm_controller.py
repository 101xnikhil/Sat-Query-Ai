import pytest
from backend.app.schemas.common import TaskType
from backend.app.schemas.trace import ToolExecutionRecord
from backend.app.controller.llm_controller import LLMReasoningController

def test_llm_controller_optical_sar_synthesis():
    """Verify LLMReasoningController accurately synthesizes Optical+SAR metrics without fabrication."""
    controller = LLMReasoningController()
    computed_metrics = {
        "water_area_km2": 16.9617,
        "water_percentage": 13.69,
        "built_up_area_km2": 9.9044,
        "built_up_percentage": 7.99,
        "cross_tool_ndwi_agreement": 0.7547
    }
    trace_steps = [
        ToolExecutionRecord(tool_name="optical_sar_fusion", status="success", duration_ms=40.0),
        ToolExecutionRecord(tool_name="spectral_index", status="success", duration_ms=15.0)
    ]

    synthesis = controller.synthesize_analysis(
        query="Provide a detailed briefing on water and urban structures",
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        tool_results={},
        computed_metrics=computed_metrics,
        trace_steps=trace_steps
    )

    assert "executive_summary" in synthesis
    assert "detailed_analysis" in synthesis
    assert "recommendations" in synthesis
    assert len(synthesis["recommendations"]) > 0

    # Ensure EXACT numbers from computed_metrics appear in the text
    exec_text = synthesis["executive_summary"]
    assert "16.9617" in exec_text or "16.96" in exec_text
    assert "9.9044" in exec_text or "9.90" in exec_text
    assert "13.69" in exec_text
    assert "7.99" in exec_text

def test_llm_controller_change_synthesis():
    """Verify LLMReasoningController correctly incorporates change dynamics."""
    controller = LLMReasoningController()
    computed_metrics = {
        "area_changed_km2": 4.5210,
        "percentage_changed": 12.45,
        "direction_of_change": "vegetation loss / deforestation"
    }

    synthesis = controller.synthesize_analysis(
        query="Analyze deforestation and land transition",
        task=TaskType.CHANGE_MAP,
        tool_results={},
        computed_metrics=computed_metrics,
        trace_steps=[]
    )

    exec_text = synthesis["executive_summary"]
    assert "4.5210" in exec_text
    assert "12.45" in exec_text
    assert "vegetation loss / deforestation" in exec_text
