import pytest
from starlette.testclient import TestClient
from backend.app.main import app
from backend.app.schemas.common import TaskType

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "rs_vqa" in data["registered_tools"]

def test_api_upload_and_query_single_vqa(optical_geotiff):
    # Upload optical image
    with open(optical_geotiff, "rb") as f:
        res = client.post("/api/upload", files={"files": ("sample_delhi.tif", f, "image/tiff")})
    assert res.status_code == 200
    upload_data = res.json()
    assert len(upload_data["images"]) == 1
    image_id = upload_data["images"][0]["image_id"]

    # Run Single VQA Query
    query_payload = {
        "query": "What is the primary terrain or land cover present?",
        "image_ids": [image_id]
    }
    q_res = client.post("/api/query", json=query_payload)
    assert q_res.status_code == 200
    res_json = q_res.json()
    assert res_json["task"] == TaskType.SINGLE_VQA.value
    assert res_json["answer"] != ""
    assert res_json["confidence_score"] > 0.5
    assert "trace" in res_json
    assert res_json["trace"]["input_validation"]["passed"] is True
    assert len(res_json["trace"]["tool_calls"]) >= 1

    # Verify trace endpoint
    session_id = res_json["session_id"]
    trace_res = client.get(f"/api/trace/{session_id}")
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    assert trace_data["session_id"] == session_id

def test_api_optical_sar_query(optical_geotiff, sar_geotiff):
    # Upload optical
    with open(optical_geotiff, "rb") as f1:
        res1 = client.post(
            "/api/upload",
            files={"files": ("delhi_opt.tif", f1, "image/tiff")},
            data={"modality": "optical"}
        )
    # Upload SAR
    with open(sar_geotiff, "rb") as f2:
        res2 = client.post(
            "/api/upload",
            files={"files": ("delhi_sar.tif", f2, "image/tiff")},
            data={"modality": "sar"}
        )
    opt_id = res1.json()["images"][0]["image_id"]
    sar_id = res2.json()["images"][0]["image_id"]

    query_payload = {
        "query": "Segment water and built-up areas using both optical and radar",
        "image_ids": [opt_id, sar_id]
    }
    q_res = client.post("/api/query", json=query_payload)
    assert q_res.status_code == 200
    res_json = q_res.json()
    assert res_json["task"] == TaskType.OPTICAL_SAR_ANALYSIS.value
    assert "water_area_km2" in res_json["computed_metrics"]
    assert "built_up_area_km2" in res_json["computed_metrics"]
    assert len(res_json["layers"]) >= 1
    assert "pdf_report_url" in res_json

def test_api_pdf_report_download(optical_geotiff, sar_geotiff):
    # Upload optical and SAR
    with open(optical_geotiff, "rb") as f1, open(sar_geotiff, "rb") as f2:
        res1 = client.post("/api/upload", files={"files": ("delhi_opt2.tif", f1, "image/tiff")})
        res2 = client.post("/api/upload", files={"files": ("delhi_sar2.tif", f2, "image/tiff")}, data={"modality": "sar"})

    opt_id = res1.json()["images"][0]["image_id"]
    sar_id = res2.json()["images"][0]["image_id"]

    q_res = client.post("/api/query", json={
        "query": "Assess built-up area and water distribution",
        "image_ids": [opt_id, sar_id]
    })
    assert q_res.status_code == 200
    session_id = q_res.json()["session_id"]

    # Download PDF report
    pdf_res = client.get(f"/api/reports/{session_id}/pdf")
    assert pdf_res.status_code == 200
    assert "application/pdf" in pdf_res.headers.get("content-type", "")
    assert pdf_res.content.startswith(b"%PDF-")
    assert len(pdf_res.content) > 1000

