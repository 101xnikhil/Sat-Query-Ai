import pytest
from models.inference.vlm_wrapper import VLMInferenceWrapper, get_vlm_wrapper
from backend.app.tools.registry import ToolRegistry
from backend.app.schemas.tools import RSVQAOutput, RSGroundingOutput

def test_vlm_wrapper_confidence_calibration():
    vlm = get_vlm_wrapper()
    # Test confidence with various logprobs
    logprobs_high = [-0.05, -0.02, -0.08]
    conf_high = vlm.calculate_token_confidence(logprobs_high)
    assert 0.85 <= conf_high <= 1.0

    logprobs_low = [-1.2, -1.5, -0.9]
    conf_low = vlm.calculate_token_confidence(logprobs_low)
    assert 0.05 <= conf_low <= 0.50

def test_vlm_wrapper_vqa(optical_geotiff):
    vlm = get_vlm_wrapper()
    res = vlm.answer_vqa("Is there vegetation present?", optical_geotiff)
    assert "answer" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "evidence" in res

def test_vlm_wrapper_grounding(optical_geotiff):
    vlm = get_vlm_wrapper()
    boxes = vlm.ground_query("Locate airport runways", optical_geotiff)
    assert len(boxes) >= 1
    for b in boxes:
        assert 0.0 <= b.ymin <= b.ymax <= 1.0
        assert 0.0 <= b.xmin <= b.xmax <= 1.0
        assert b.confidence > 0.0

def test_real_rs_vqa_tool_via_registry(optical_geotiff):
    reg = ToolRegistry.get_instance()
    out: RSVQAOutput = reg.execute("rs_vqa", {
        "query": "Describe the water bodies in this scene",
        "image_path": optical_geotiff,
        "confidence_threshold": 0.5
    })
    assert isinstance(out, RSVQAOutput)
    assert out.confidence > 0.0
    assert out.answer != ""

def test_real_rs_grounding_tool_via_registry(optical_geotiff):
    reg = ToolRegistry.get_instance()
    out: RSGroundingOutput = reg.execute("rs_grounding", {
        "query": "Where are the urban clusters?",
        "image_path": optical_geotiff
    })
    assert isinstance(out, RSGroundingOutput)
    assert out.detected_count >= 1
    assert len(out.geojson.features) == out.detected_count
    assert out.confidences[0] > 0.0
