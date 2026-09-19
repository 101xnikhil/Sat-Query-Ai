import os
import pytest
import numpy as np
import rasterio
from models.inference.change_detector import compute_radiometric_change
from backend.app.tools.registry import ToolRegistry
from backend.app.schemas.tools import ChangeMapOutput, ChangeVQAOutput

def test_change_detection_identical_images(optical_geotiff):
    res = compute_radiometric_change(
        image_t1_path=optical_geotiff,
        image_t2_path=optical_geotiff,
        threshold=0.45
    )
    assert res["area_changed_km2"] == 0.0
    assert res["percentage_changed"] == 0.0
    assert "no significant change" in res["direction_of_change"]
    assert os.path.exists(res["change_mask_path"])

def test_change_detection_modified_images(test_data_dir, optical_geotiff):
    # Create modified T2 image with a block of high values
    t2_path = os.path.join(test_data_dir, "t2_modified.tif")
    with rasterio.open(optical_geotiff) as src:
        profile = src.profile.copy()
        data = src.read()
        # Introduce change in a 30x30 block
        data[:, 20:50, 20:50] = 250

    with rasterio.open(t2_path, 'w', **profile) as dst:
        dst.write(data)

    res = compute_radiometric_change(
        image_t1_path=optical_geotiff,
        image_t2_path=t2_path,
        threshold=0.30
    )
    assert res["area_changed_km2"] > 0.0
    assert res["percentage_changed"] > 0.0
    assert any(term in res["direction_of_change"] for term in ["vegetation loss", "construction", "brightness"])
    assert res["geojson"] is not None
    assert len(res["geojson"].features) >= 1

    # Verify georeferencing of generated change mask
    with rasterio.open(res["change_mask_path"]) as mask_src:
        assert mask_src.crs.to_string() == "EPSG:4326"
        assert mask_src.width == src.width
        assert mask_src.height == src.height

def test_real_change_map_tool_via_registry(optical_geotiff):
    reg = ToolRegistry.get_instance()
    out: ChangeMapOutput = reg.execute("change_map", {
        "image_t1_path": optical_geotiff,
        "image_t2_path": optical_geotiff,
        "threshold": 0.5
    })
    assert isinstance(out, ChangeMapOutput)
    assert out.area_changed_km2 == 0.0
    assert out.percentage_changed == 0.0

def test_real_change_vqa_tool_via_registry(optical_geotiff):
    reg = ToolRegistry.get_instance()
    out: ChangeVQAOutput = reg.execute("change_vqa", {
        "query": "Did any significant changes occur?",
        "image_t1_path": optical_geotiff,
        "image_t2_path": optical_geotiff
    })
    assert isinstance(out, ChangeVQAOutput)
    assert "no" in out.answer.lower()
    assert out.confidence > 0.5
