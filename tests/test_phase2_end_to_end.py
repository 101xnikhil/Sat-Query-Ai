import pytest
from starlette.testclient import TestClient
from backend.app.main import app
from backend.app.tools.registry import ToolRegistry
from backend.app.schemas.tools import RSVQAInput, RSGroundingInput, RSCaptionInput

client = TestClient(app)

def test_rs_vqa_tool_phase2(optical_geotiff):
    reg = ToolRegistry.get_instance()
    tool = reg.get("rs_vqa")
    out = tool.run(RSVQAInput(query="Is there a water body?", image_path=optical_geotiff))
    assert out.mode == "model"
    assert out.confidence > 0.70
    assert len(out.tiles_used) >= 1
    assert out.rendering_applied in ["multispectral_true_color_rgb", "false_color_cir", "sar_db_grayscale"]

def test_rs_grounding_tool_phase2(optical_geotiff):
    reg = ToolRegistry.get_instance()
    tool = reg.get("rs_grounding")
    out = tool.run(RSGroundingInput(query="highlight the water body", image_path=optical_geotiff))
    assert out.mode == "model"
    assert out.detected_count >= 1
    assert len(out.geojson.features) == out.detected_count
    assert len(out.tiles_used) >= 1
    assert out.rendering_applied is not None

    # Check that counts and areas come strictly from detections/masks
    for feat in out.geojson.features:
        assert "area_m2" in feat.properties
        assert "area_pixels" in feat.properties
        assert feat.properties["area_m2"] > 0
        assert feat.properties["score"] >= 0.30

def test_rs_caption_tool_phase2(optical_geotiff):
    reg = ToolRegistry.get_instance()
    tool = reg.get("rs_caption")
    out = tool.run(RSCaptionInput(image_path=optical_geotiff))
    assert out.mode == "model"
    assert len(out.caption) > 10
    assert len(out.tags) >= 2
    assert out.confidence > 0.70

def test_end_to_end_api_highlight_water_body(optical_geotiff):
    """
    Acceptance Test:
    Upload image + query 'highlight the water body'
    Produces a visible overlay layer, task='grounding', and execution trace showing rs_grounding with parameters.
    """
    with open(optical_geotiff, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"files": ("delhi_optical.tif", f, "image/tiff")},
            data={"modality_override": "optical"}
        )
    assert upload_resp.status_code == 200
    images = upload_resp.json()["images"]
    assert len(images) == 1
    session_id = f"session_{images[0]['image_id']}"

    # Run Query
    query_resp = client.post("/api/query", json={
        "session_id": session_id,
        "query": "highlight the water body",
        "image_ids": [img["image_id"] for img in images]
    })
    assert query_resp.status_code == 200
    data = query_resp.json()

    assert data["task"] == "grounding"
    assert "water" in data["answer"].lower()
    assert data["confidence_score"] > 0.70

    # Verify overlay layer
    assert len(data["layers"]) >= 1
    layer = data["layers"][0]
    assert layer["type"] == "geojson"
    assert layer["geojson"]["type"] == "FeatureCollection"
    assert len(layer["geojson"]["features"]) >= 1

    # Verify trace includes rs_grounding with parameters and rendering/tiling
    trace = data["trace"]
    tool_calls = trace["tool_calls"]
    assert any(t["tool_name"] == "rs_grounding" for t in tool_calls)
    grounding_call = next(t for t in tool_calls if t["tool_name"] == "rs_grounding")
    assert "query" in grounding_call["parameters"]
    assert grounding_call["parameters"]["query"] == "highlight the water body"
    assert "rendering_applied" in grounding_call["output_summary"]
    assert "tiles_used" in grounding_call["output_summary"]
