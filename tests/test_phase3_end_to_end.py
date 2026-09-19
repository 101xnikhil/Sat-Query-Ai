import os
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import get_settings

client = TestClient(app)

def test_phase3_acceptance_what_changed_and_where(synthetic_bitemporal_square_pair):
    """
    Acceptance Criteria Test 1:
    Upload two dates, ask 'what changed between these two dates, and where?' and get:
    - a description
    - change map overlay (GeoJSON vector layer in #f43f5e)
    - area statistics (m², ha, km², percentage of scene, direction)
    - trace showing both tools in order (change_map then change_vqa)
    """
    meta1, meta2 = synthetic_bitemporal_square_pair

    # 1. Upload two dates via API
    with open(meta1.file_path, "rb") as f1, open(meta2.file_path, "rb") as f2:
        res1 = client.post(
            "/api/upload",
            files=[("files", ("date1_t1.tif", f1, "image/tiff"))],
            data={"acquisition_date": "2023-01-10T10:00:00Z"}
        )
        res2 = client.post(
            "/api/upload",
            files=[("files", ("date2_t2.tif", f2, "image/tiff"))],
            data={"acquisition_date": "2023-06-15T10:00:00Z"}
        )

    assert res1.status_code == 200, res1.text
    assert res2.status_code == 200, res2.text
    id1 = res1.json()["images"][0]["image_id"]
    id2 = res2.json()["images"][0]["image_id"]

    # 2. Run query: "What changed between these two dates, and where?"
    query_payload = {
        "query": "What changed between these two dates, and where?",
        "image_ids": [id1, id2]
    }
    resp = client.post("/api/query", json=query_payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Verify Response Structure
    assert data["session_id"] is not None
    assert data["task"] in ["change_vqa", "change_map"]
    assert "40000.0 m²" in data["answer"] or "4.000 ha" in data["answer"] or "4.00%" in data["answer"]

    # Verify Area Statistics
    metrics = data["computed_metrics"]
    assert "area_changed_m2" in metrics
    assert "area_changed_ha" in metrics
    assert "area_changed_km2" in metrics
    assert "percentage_changed" in metrics
    assert "direction_of_change" in metrics
    assert metrics["area_changed_m2"] == 40000.0
    assert metrics["area_changed_ha"] == 4.0
    assert metrics["percentage_changed"] == 4.0

    # Verify Change Map Overlay Layer
    assert len(data["layers"]) >= 1
    change_layer = next((l for l in data["layers"] if "change" in l["name"].lower() or "temporal" in l["name"].lower()), None)
    assert change_layer is not None
    assert change_layer["type"] == "geojson"
    assert change_layer["geojson"] is not None
    assert len(change_layer["geojson"]["features"]) >= 1

    # Verify Execution Trace: showing both tools in order!
    trace = data["trace"]
    tool_calls = trace["tool_calls"]
    tool_names = [tc["tool_name"] for tc in tool_calls]
    assert "change_map" in tool_names
    assert "change_vqa" in tool_names

    # Verify order: change_map first, then change_vqa
    idx_map = tool_names.index("change_map")
    idx_vqa = tool_names.index("change_vqa")
    assert idx_map < idx_vqa, f"change_map must precede change_vqa in trace, got {tool_names}"

    # Verify trace parameters passed from change_map to change_vqa
    vqa_call = tool_calls[idx_vqa]
    assert "change_mask_path" in vqa_call["parameters"]
    assert "change_stats" in vqa_call["parameters"]
    assert vqa_call["parameters"]["change_stats"]["area_changed_m2"] == 40000.0

def test_phase3_acceptance_directional_question(synthetic_bitemporal_square_pair):
    """
    Acceptance Criteria Test 2:
    'Has built-up area increased, decreased, or remained unchanged?'
    Returns a mask-derived answer with confidence.
    """
    meta1, meta2 = synthetic_bitemporal_square_pair

    # Upload images
    with open(meta1.file_path, "rb") as f1, open(meta2.file_path, "rb") as f2:
        res1 = client.post(
            "/api/upload",
            files=[("files", ("t1.tif", f1, "image/tiff"))],
            data={"acquisition_date": "2023-01-01T00:00:00Z"}
        )
        res2 = client.post(
            "/api/upload",
            files=[("files", ("t2.tif", f2, "image/tiff"))],
            data={"acquisition_date": "2023-05-01T00:00:00Z"}
        )

    assert res1.status_code == 200, res1.text
    assert res2.status_code == 200, res2.text
    id1 = res1.json()["images"][0]["image_id"]
    id2 = res2.json()["images"][0]["image_id"]

    query_payload = {
        "query": "Has built-up area increased, decreased, or remained unchanged?",
        "image_ids": [id1, id2]
    }
    resp = client.post("/api/query", json=query_payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Must return mask-derived direction
    assert "increased" in data["answer"].lower()
    assert data["confidence_score"] >= 0.80
    assert data["computed_metrics"]["area_changed_m2"] > 0
    assert data["pdf_report_url"] is not None
