import pytest
from backend.app.schemas.common import Modality, TaskType, ImageMetadata, GeoBBox, RasterFormat
from backend.app.config import get_settings
from backend.ingest.reader import read_geotiff_metadata
from backend.validators.results import ValidatorResult
from backend.validators.validators import (
    validate_image_count,
    validate_modality_compatibility,
    validate_crs_and_overlap,
    validate_acquisition_dates,
    validate_band_count_and_dtype,
    validate_and_preprocess_sar,
    validate_all
)

def test_validate_image_count_compatible(optical_geotiff):
    meta = read_geotiff_metadata(optical_geotiff)
    # Single image task with 1 image
    res = validate_image_count([meta], TaskType.SINGLE_VQA)
    assert res.ok is True
    assert len(res.errors) == 0
    assert len(res.actions_taken) >= 1

    # Dual image task with 2 images
    res_dual = validate_image_count([meta, meta], TaskType.CHANGE_VQA)
    assert res_dual.ok is True
    assert len(res_dual.errors) == 0

def test_validate_image_count_incompatible(optical_geotiff):
    meta = read_geotiff_metadata(optical_geotiff)
    # Single image task with 2 images
    res = validate_image_count([meta, meta], TaskType.SINGLE_VQA)
    assert res.ok is False
    assert any("requires exactly 1 image" in e for e in res.errors)

    # Dual image task with 1 image
    res_dual = validate_image_count([meta], TaskType.CHANGE_MAP)
    assert res_dual.ok is False
    assert any("requires exactly 2 images" in e for e in res_dual.errors)

def test_validate_modality_compatibility_compatible(optical_geotiff, sar_geotiff):
    opt_meta = read_geotiff_metadata(optical_geotiff)
    sar_meta = read_geotiff_metadata(sar_geotiff)

    res = validate_modality_compatibility([opt_meta, sar_meta], TaskType.OPTICAL_SAR_ANALYSIS)
    assert res.ok is True
    assert len(res.errors) == 0
    assert any("cross-modal" in a.lower() for a in res.actions_taken)

def test_validate_modality_compatibility_incompatible(optical_geotiff):
    opt1 = read_geotiff_metadata(optical_geotiff)
    opt2 = read_geotiff_metadata(optical_geotiff)

    # Optical + Optical for Optical-SAR task is incompatible
    res = validate_modality_compatibility([opt1, opt2], TaskType.OPTICAL_SAR_ANALYSIS)
    assert res.ok is False
    assert any("requires one Optical/Multispectral image and one SAR image" in e for e in res.errors)

def test_validate_crs_and_overlap_compatible(optical_geotiff, sar_geotiff):
    meta1 = read_geotiff_metadata(optical_geotiff)
    meta2 = read_geotiff_metadata(sar_geotiff)

    res, processed = validate_crs_and_overlap([meta1, meta2], min_overlap_iou=0.10)
    assert res.ok is True
    assert len(res.errors) == 0
    assert any("Spatial extent overlap verified" in a for a in res.actions_taken)

def test_validate_crs_and_overlap_incompatible(optical_geotiff, disjoint_geotiff):
    meta1 = read_geotiff_metadata(optical_geotiff)
    meta2 = read_geotiff_metadata(disjoint_geotiff)

    res, processed = validate_crs_and_overlap([meta1, meta2], min_overlap_iou=0.10)
    assert res.ok is False
    assert any("below minimum threshold" in e for e in res.errors)

def test_validate_crs_auto_reproject(optical_geotiff, utm_reproject_geotiff, test_data_dir):
    meta_wgs84 = read_geotiff_metadata(optical_geotiff)
    meta_utm = read_geotiff_metadata(utm_reproject_geotiff)

    assert meta_wgs84.crs != meta_utm.crs

    res, processed = validate_crs_and_overlap(
        [meta_wgs84, meta_utm],
        min_overlap_iou=0.01,
        auto_reproject=True,
        output_dir=test_data_dir
    )
    assert res.ok is True
    assert any("Auto-reprojected" in a for a in res.actions_taken)
    assert processed[1].crs == meta_wgs84.crs

def test_validate_acquisition_dates_ordered(temporal_geotiffs_with_dates):
    p1, p2 = temporal_geotiffs_with_dates
    m1 = read_geotiff_metadata(p1)
    m2 = read_geotiff_metadata(p2)

    res, processed = validate_acquisition_dates([m1, m2], TaskType.CHANGE_VQA)
    assert res.ok is True
    assert len(res.errors) == 0
    assert any("Verified chronological order" in a for a in res.actions_taken)
    assert processed[0].acquisition_date < processed[1].acquisition_date

def test_validate_acquisition_dates_reversed(temporal_geotiffs_with_dates):
    p1, p2 = temporal_geotiffs_with_dates
    m1 = read_geotiff_metadata(p1)  # 2023-01-15
    m2 = read_geotiff_metadata(p2)  # 2023-06-20

    # Pass reversed: m2 (later) first, m1 (earlier) second
    res, processed = validate_acquisition_dates([m2, m1], TaskType.CHANGE_VQA)
    assert res.ok is True
    assert any("Auto-reordered bi-temporal images chronologically" in a for a in res.actions_taken)
    # First image in processed should now be the earlier date
    assert processed[0].acquisition_date < processed[1].acquisition_date

def test_validate_acquisition_dates_missing(optical_geotiff):
    m1 = read_geotiff_metadata(optical_geotiff)
    m2 = read_geotiff_metadata(optical_geotiff)
    m1.acquisition_date = None
    m2.acquisition_date = None

    res, processed = validate_acquisition_dates([m1, m2], TaskType.CHANGE_VQA)
    assert res.ok is True
    assert any("Acquisition date missing" in w for w in res.warnings)

def test_validate_band_count_and_dtype_compatible(optical_geotiff):
    meta = read_geotiff_metadata(optical_geotiff)
    res = validate_band_count_and_dtype([meta])
    assert res.ok is True
    assert any("Band count and dtype sanity verified" in a for a in res.actions_taken)

def test_validate_band_count_and_dtype_incompatible():
    bad_meta = ImageMetadata(
        image_id="corrupt_img",
        filename="corrupt.tif",
        file_path="/tmp/corrupt.tif",
        format=RasterFormat.GEOTIFF,
        modality=Modality.OPTICAL,
        width=0,
        height=0,
        band_count=0
    )
    res = validate_band_count_and_dtype([bad_meta])
    assert res.ok is False
    assert any("invalid band count" in e for e in res.errors)

def test_validate_and_preprocess_sar(sar_geotiff, test_data_dir):
    sar_meta = read_geotiff_metadata(sar_geotiff)
    res, processed = validate_and_preprocess_sar(
        [sar_meta],
        db_conversion=True,
        speckle_filter="lee",
        window_size=3,
        output_dir=test_data_dir
    )
    assert res.ok is True
    assert any("SAR preprocessing" in a for a in res.actions_taken)
    assert any("Linear-to-dB" in a for a in res.actions_taken)
    assert any("Lee speckle filter" in a for a in res.actions_taken)
    assert processed[0].image_id.startswith("preprocessed_")

def test_validate_all_pipeline(optical_geotiff, sar_geotiff, test_data_dir):
    settings = get_settings()
    opt_meta = read_geotiff_metadata(optical_geotiff)
    sar_meta = read_geotiff_metadata(sar_geotiff)

    res, rec, processed = validate_all(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        images=[opt_meta, sar_meta],
        settings=settings,
        output_dir=test_data_dir
    )
    assert res.ok is True
    assert rec.passed is True
    assert len(res.actions_taken) >= 3
