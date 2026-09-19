import os
from datetime import datetime
from pathlib import Path
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.app.schemas.common import Modality, TaskType, ImageMetadata
from backend.validators.validators import (
    validate_crs_and_overlap,
    validate_acquisition_dates,
    reproject_raster_to_match,
    compute_extent_overlap
)
from models.inference.change_detector import compute_radiometric_change

def test_synthetic_area_statistics_exact_match(synthetic_bitemporal_square_pair):
    """Verifies that the synthetic 20x20 square produces exactly 40,000 m² and 4.0% change."""
    meta1, meta2 = synthetic_bitemporal_square_pair
    res = compute_radiometric_change(
        image_t1_path=meta1.file_path,
        image_t2_path=meta2.file_path,
        threshold=0.30,
        min_area_m2=100.0
    )

    # Exact known area calculations:
    # 400 pixels * 100 m² = 40,000 m²
    assert res["changed_pixel_count"] == 400
    assert abs(res["area_changed_m2"] - 40000.0) < 1.0
    assert abs(res["area_changed_ha"] - 4.0) < 0.01
    assert abs(res["area_changed_km2"] - 0.04) < 0.001
    assert abs(res["percentage_changed"] - 4.00) < 0.05
    assert res["geojson"] is not None
    assert len(res["geojson"].features) >= 1
    # Verify polygon property contains area
    assert res["geojson"].features[0].properties["area_m2"] > 0

def test_date_ordering_and_auto_swap(synthetic_bitemporal_square_pair):
    """Verifies chronological ordering and automatic chronological swap when dates are inverted."""
    meta1, meta2 = synthetic_bitemporal_square_pair

    # 1. Normal order: T1 (Jan 2023) < T2 (June 2023)
    val_res, ordered = validate_acquisition_dates([meta1, meta2], TaskType.CHANGE_VQA)
    assert val_res.ok
    assert ordered[0].image_id == "synthetic_t1"
    assert ordered[1].image_id == "synthetic_t2"
    assert any("Verified chronological order" in act for act in val_res.actions_taken)

    # 2. Inverted order: pass T2 first, T1 second -> should auto-swap!
    val_res_inv, ordered_inv = validate_acquisition_dates([meta2, meta1], TaskType.CHANGE_VQA)
    assert val_res_inv.ok
    assert ordered_inv[0].image_id == "synthetic_t1"  # Swapped back to earlier date
    assert ordered_inv[1].image_id == "synthetic_t2"
    assert any("Auto-reordered bi-temporal images chronologically" in act for act in val_res_inv.actions_taken)

    # 3. Missing date warning
    meta_no_date = meta1.model_copy()
    meta_no_date.acquisition_date = None
    val_res_missing, _ = validate_acquisition_dates([meta_no_date, meta2], TaskType.CHANGE_VQA)
    assert val_res_missing.ok
    assert any("Acquisition date missing" in w for w in val_res_missing.warnings)

    # 4. Identical dates warning
    meta_same_date = meta2.model_copy()
    meta_same_date.acquisition_date = meta1.acquisition_date
    val_res_same, _ = validate_acquisition_dates([meta1, meta_same_date], TaskType.CHANGE_VQA)
    assert val_res_same.ok
    assert any("identical acquisition dates" in w for w in val_res_same.warnings)

def test_misalignment_and_auto_alignment(tmp_path, synthetic_bitemporal_square_pair):
    """
    Tests handling of spatial misalignment:
    Creates a resampled/misaligned T2 (different resolution and pixel grid dimensions)
    and verifies that validate_spatial_extent_and_coregistration reprojects to the common grid.
    """
    meta1, meta2 = synthetic_bitemporal_square_pair

    # Create a misaligned T2: 50x50 pixels with 20m resolution instead of 100x100 10m
    misaligned_path = str(tmp_path / "misaligned_t2.tif")
    transform_20m = from_origin(77.0, 28.0, 20.0 / 111320.0, 20.0 / 111320.0)
    data_50x50 = np.full((4, 50, 50), 180, dtype=np.uint8)

    with rasterio.open(
        misaligned_path, 'w',
        driver='GTiff',
        height=50, width=50,
        count=4, dtype=rasterio.uint8,
        crs="EPSG:4326", transform=transform_20m
    ) as dst:
        dst.write(data_50x50)

    from backend.app.ingestion.reader import read_image_metadata
    meta2_misaligned = read_image_metadata(
        misaligned_path,
        image_id="misaligned_t2",
        acquisition_date="2023-06-20T10:00:00Z"
    )

    # Validate and auto-align to common grid
    val_res, aligned_images = validate_crs_and_overlap(
        images=[meta1, meta2_misaligned],
        auto_reproject=True,
        output_dir=str(tmp_path / "aligned")
    )

    assert val_res.ok
    assert any("auto-aligned" in act.lower() for act in val_res.actions_taken)
    # The second image should now be aligned to 100x100 matching img1
    aligned_meta2 = aligned_images[1]
    assert aligned_meta2.width == meta1.width
    assert aligned_meta2.height == meta1.height
    assert Path(aligned_meta2.file_path).exists()
