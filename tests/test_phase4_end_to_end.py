import os
import pytest
from fastapi.testclient import TestClient
import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.app.main import app
from backend.app.controller.engine import ControllerEngine
from backend.app.schemas.query import QueryRequest
from backend.app.schemas.common import TaskType
from models.data.optical_sar import OpticalSARDataLoader

client = TestClient(app)

@pytest.fixture
def optical_sar_fixture(tmp_path):
    return OpticalSARDataLoader.create_synthetic_fixture(
        output_dir=tmp_path / "e2e_fixtures",
        sample_id="e2e_pair_01",
        height=100,
        width=100,
        pixel_size_m=10.0
    )

def test_phase4_end_to_end_api(optical_sar_fixture):
    """
    Full Acceptance Test:
    - Upload optical+SAR pair
    - Query: 'use the optical and SAR images together to identify built-up and water regions'
    - Get masks, areas (m2, ha, km2, %), agreement score, and trace showing fusion and index tools.
    """
    req_payload = {
        "query": "use the optical and SAR images together to identify built-up and water regions",
        "image_ids": [optical_sar_fixture["optical_path"], optical_sar_fixture["sar_path"]]
    }
    response = client.post("/api/query", json=req_payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["task"] == "optical_sar_analysis"
    assert "Surface Water:" in data["answer"]
    assert "Built-up Urban:" in data["answer"]
    assert "m²" in data["answer"]
    assert "ha" in data["answer"]

    # Verify metrics
    metrics = data["computed_metrics"]
    assert "water_area_m2" in metrics
    assert "water_area_ha" in metrics
    assert "built_up_area_m2" in metrics
    assert "built_up_area_ha" in metrics
    assert "cross_tool_ndwi_agreement" in metrics
    assert metrics["cross_tool_ndwi_agreement"] >= 0.50

    # Verify layers for GUI toggles
    layers = data["layers"]
    layer_names = [l["name"] for l in layers]
    assert any("Water Mask" in n for n in layer_names)
    assert any("Built-up Mask" in n for n in layer_names)

    # Verify trace shows sequential tools
    trace = data["trace"]
    tool_calls = [tc["tool_name"] for tc in trace["tool_calls"]]
    assert "optical_sar_fusion" in tool_calls
    assert "spectral_index" in tool_calls

    # Verify PDF report generation endpoint
    session_id = data["session_id"]
    pdf_resp = client.get(f"/api/reports/{session_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"

def test_phase4_missing_nir_degrades_gracefully(tmp_path):
    """
    Acceptance Test:
    Missing NIR band in optical image degrades gracefully without failing.
    """
    rgb_path = tmp_path / "optical_rgb_only.tif"
    sar_path = tmp_path / "sar_dual.tif"
    h, w = 60, 60
    transform = from_origin(77.5946, 12.9716, 0.0001, 0.0001)

    # 3-band RGB optical (no NIR)
    with rasterio.open(rgb_path, "w", driver="GTiff", height=h, width=w, count=3, dtype=rasterio.float32, crs="EPSG:4326", transform=transform) as dst:
        dst.write(np.random.uniform(0.1, 0.8, (3, h, w)).astype(np.float32))

    # 2-band SAR
    with rasterio.open(sar_path, "w", driver="GTiff", height=h, width=w, count=2, dtype=rasterio.float32, crs="EPSG:4326", transform=transform) as dst:
        dst.write(np.random.uniform(-25.0, 5.0, (2, h, w)).astype(np.float32))

    req_payload = {
        "query": "use the optical and SAR images together to identify built-up and water regions",
        "image_ids": [str(rgb_path), str(sar_path)]
    }
    response = client.post("/api/query", json=req_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["task"] == "optical_sar_analysis"
    # NDWI not applicable should be noted in answer and metrics without crash
    assert "not applicable" in data["answer"].lower() or "missing" in data["answer"].lower()
    assert data["computed_metrics"].get("cross_tool_ndwi_status") == "not_applicable"
