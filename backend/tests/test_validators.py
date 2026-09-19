import os
import pytest
import numpy as np
from backend.app.validators.spatial import compute_extent_overlap, reproject_raster_to_match
from backend.app.validators.sar import convert_to_db, lee_speckle_filter, preprocess_sar_image
from backend.app.validators.pipeline import validate_task_inputs
from backend.app.ingestion.reader import read_image_metadata
from backend.app.schemas.common import Modality, TaskType
from backend.app.config import get_settings

def test_spatial_overlap_identical(optical_geotiff, sar_geotiff):
    meta1 = read_image_metadata(optical_geotiff)
    meta2 = read_image_metadata(sar_geotiff)
    iou, meta = compute_extent_overlap(meta1, meta2)
    assert iou == pytest.approx(1.0, rel=1e-2)

def test_spatial_overlap_disjoint(optical_geotiff, disjoint_geotiff):
    meta1 = read_image_metadata(optical_geotiff)
    meta2 = read_image_metadata(disjoint_geotiff)
    iou, meta = compute_extent_overlap(meta1, meta2)
    assert iou == 0.0

def test_sar_db_and_lee_filter():
    raw_sar = np.array([[10.0, 50.0, 10.0], [50.0, 200.0, 50.0], [10.0, 50.0, 10.0]], dtype=np.float32)
    db_arr = convert_to_db(raw_sar)
    assert db_arr.shape == raw_sar.shape
    # 10 * log10(10) = 10 dB
    assert db_arr[0, 0] == pytest.approx(10.0, rel=1e-2)

    filtered = lee_speckle_filter(raw_sar, window_size=3)
    assert filtered.shape == raw_sar.shape
    # Center spike should be smoothed
    assert filtered[1, 1] < raw_sar[1, 1]

def test_auto_reproject_utm_to_wgs84(optical_geotiff, utm_reproject_geotiff, test_data_dir):
    out_path = os.path.join(test_data_dir, "reprojected_test.tif")
    res = reproject_raster_to_match(
        source_path=utm_reproject_geotiff,
        reference_path=optical_geotiff,
        output_path=out_path
    )
    assert os.path.exists(out_path)
    assert res["dst_crs"] == "EPSG:4326"

def test_pipeline_validation_image_count(optical_geotiff, sar_geotiff):
    settings = get_settings()
    meta_opt = read_image_metadata(optical_geotiff)
    meta_sar = read_image_metadata(sar_geotiff)

    # single_vqa expects 1 image, passing 2 should fail
    passed, record, _ = validate_task_inputs(TaskType.SINGLE_VQA, [meta_opt, meta_sar], settings)
    assert passed is False
    assert any("requires exactly 1 image" in e for e in record.errors)

def test_pipeline_validation_optical_sar_modality(optical_geotiff):
    settings = get_settings()
    meta1 = read_image_metadata(optical_geotiff)
    meta2 = read_image_metadata(optical_geotiff)

    # optical_sar_analysis expects 1 optical and 1 SAR. Passing two optical should fail!
    passed, record, _ = validate_task_inputs(TaskType.OPTICAL_SAR_ANALYSIS, [meta1, meta2], settings)
    assert passed is False
    assert any("requires one Optical/Multispectral image and one SAR image" in e for e in record.errors)

def test_pipeline_validation_disjoint_rejection(optical_geotiff, disjoint_geotiff):
    settings = get_settings()
    meta_opt = read_image_metadata(optical_geotiff)
    meta_disjoint = read_image_metadata(disjoint_geotiff)

    passed, record, _ = validate_task_inputs(TaskType.CHANGE_MAP, [meta_opt, meta_disjoint], settings)
    assert passed is False
    assert any("below minimum threshold" in e for e in record.errors)
