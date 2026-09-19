import pytest
from backend.app.schemas.common import Modality, TaskType, ImageMetadata
from backend.app.controller.llm_planner import LLMPlanner, ToolCallPlan, LLMPlanOutput

@pytest.fixture
def dummy_optical_image():
    return ImageMetadata(
        image_id="opt_sample",
        filename="opt_sample.tif",
        file_path="data/opt_sample.tif",
        format="geotiff",
        width=256,
        height=256,
        crs="EPSG:4326",
        band_count=3,
        dtype="uint8",
        resolution=(10.0, 10.0),
        modality=Modality.OPTICAL,
        modality_confidence=0.98
    )

def test_planner_clean_json_removes_chain_of_thought():
    """Verify chain-of-thought <think> tags and markdown are completely stripped."""
    planner = LLMPlanner()
    raw = (
        "<think>The user wants to find runways. I will call rs_grounding with query.</think>\n"
        "```json\n"
        '{"task": "grounding", "tool_calls": [{"tool_name": "rs_grounding", "parameters": {"query": "runway"}}]}\n'
        "```"
    )
    cleaned = planner._clean_json_output(raw)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")

def test_planner_falls_back_on_invalid_json(dummy_optical_image):
    """Verify fallback occurs and fallback_used=True when JSON is unparseable."""
    planner = LLMPlanner()
    # Force mock output that has broken syntax
    planner._generate_local_instruction_plan_json = lambda q, img, t: "INVALID JSON NOT BRACKETED"
    task, plan, fallback_used, actions = planner.plan(
        query="Is there an airport here?",
        images=[dummy_optical_image]
    )
    assert fallback_used is True
    assert task == TaskType.SINGLE_VQA
    assert any("fallback" in a.lower() for a in actions)
    assert len(plan) > 0
    assert plan[0][0] == "rs_vqa"

def test_planner_clamps_and_strips_parameters(dummy_optical_image):
    """Verify parameter whitelist enforcement clamps out-of-bounds parameters and strips unwhitelisted ones."""
    planner = LLMPlanner()
    tool_calls = [
        ToolCallPlan(
            tool_name="rs_vqa",
            parameters={
                "query": "Is there a lake?",
                "confidence_threshold": 99.5,  # Exceeds 1.0, should clamp to 1.0
                "malicious_injection": "DROP TABLE users;",  # Not in whitelist, should be stripped
                "unauthorized_flag": True
            }
        )
    ]
    validated, violations = planner._enforce_whitelist_and_clamp(tool_calls, [dummy_optical_image])
    assert len(validated) == 1
    sanitized = validated[0].parameters
    assert sanitized["confidence_threshold"] == 1.0
    assert "malicious_injection" not in sanitized
    assert "unauthorized_flag" not in sanitized
    assert any("stripped" in v for v in violations)
    assert any("Clamped" in v for v in violations)

def test_planner_falls_back_when_all_tools_unregistered(dummy_optical_image):
    """Verify fallback when an adversarial prompt tries to call a completely unregistered tool."""
    planner = LLMPlanner()
    tool_calls = [
        ToolCallPlan(
            tool_name="bash_exec_unregistered",
            parameters={"command": "whoami"}
        )
    ]
    validated, violations = planner._enforce_whitelist_and_clamp(tool_calls, [dummy_optical_image])
    assert len(validated) == 0
    assert any("Registry violation" in v for v in violations)
