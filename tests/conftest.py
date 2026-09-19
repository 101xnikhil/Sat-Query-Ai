import os
import shutil
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

# Re-export session fixtures from backend/tests/conftest.py
from backend.tests.conftest import (
    test_data_dir,
    optical_geotiff,
    sar_geotiff,
    utm_reproject_geotiff,
    disjoint_geotiff,
    benchmark_png
)

@pytest.fixture(scope="session")
def temporal_geotiffs_with_dates(test_data_dir):
    """
    Generates two GeoTIFFs with embedded acquisition date tags:
    T1: 2023-01-15 (pre-event)
    T2: 2023-06-20 (post-event)
    """
    path1 = os.path.join(test_data_dir, "temporal_t1_20230115.tif")
    path2 = os.path.join(test_data_dir, "temporal_t2_20230620.tif")
    width, height = 80, 80
    minx, miny, maxx, maxy = 77.0, 28.0, 77.04, 28.04
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    np.random.seed(101)
    data1 = np.random.randint(50, 150, (3, height, width), dtype=np.uint8)
    data2 = np.random.randint(70, 180, (3, height, width), dtype=np.uint8)

    with rasterio.open(
        path1, 'w', driver='GTiff', height=height, width=width, count=3,
        dtype=rasterio.uint8, crs='EPSG:4326', transform=transform
    ) as dst1:
        dst1.write(data1)
        dst1.update_tags(TIFFTAG_DATETIME="2023:01:15 10:00:00", ACQUISITION_DATE="2023-01-15T10:00:00Z")

    with rasterio.open(
        path2, 'w', driver='GTiff', height=height, width=width, count=3,
        dtype=rasterio.uint8, crs='EPSG:4326', transform=transform
    ) as dst2:
        dst2.write(data2)
        dst2.update_tags(TIFFTAG_DATETIME="2023:06:20 10:00:00", ACQUISITION_DATE="2023-06-20T10:00:00Z")

    return path1, path2

@pytest.fixture
def synthetic_bitemporal_square_pair(test_data_dir):
    """
    Creates a synthetic bi-temporal pair with known geometric area:
    Grid: 100 x 100 pixels, resolution 10.0m x 10.0m (pixel area = 100 m²).
    T1: uniform background (mean 50).
    T2: background 50 + a 20x20 square (from row 30:50, col 30:50) with value 220.
    Known changed area = 20 * 20 = 400 pixels = 40,000 m² = 4.0 ha = 0.04 km².
    Scene total area = 100 * 100 = 10,000 pixels = 1,000,000 m² = 100 ha = 1.0 km².
    Percentage changed = 400 / 10000 = 4.00%.
    """
    from rasterio.transform import from_origin
    from backend.app.schemas.common import Modality, ImageMetadata

    width, height = 100, 100
    res = 10.0
    transform = from_origin(77.0, 28.0, res / 111320.0, res / 111320.0)
    crs = "EPSG:4326"

    # T1 array: 4-band image (B, G, R, NIR)
    t1_data = np.full((4, height, width), 50, dtype=np.uint8)
    t1_path = os.path.join(test_data_dir, "synthetic_t1.tif")

    with rasterio.open(
        t1_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=4, dtype=rasterio.uint8,
        crs=crs, transform=transform
    ) as dst:
        dst.write(t1_data)

    # T2 array: identical background, but with a 20x20 high-intensity square added
    t2_data = np.full((4, height, width), 50, dtype=np.uint8)
    t2_data[:, 30:50, 30:50] = 220
    t2_path = os.path.join(test_data_dir, "synthetic_t2.tif")

    with rasterio.open(
        t2_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=4, dtype=rasterio.uint8,
        crs=crs, transform=transform
    ) as dst:
        dst.write(t2_data)

    from backend.app.ingestion.reader import read_image_metadata

    meta1 = read_image_metadata(t1_path, image_id="synthetic_t1", acquisition_date="2023-01-15T10:00:00Z")
    meta2 = read_image_metadata(t2_path, image_id="synthetic_t2", acquisition_date="2023-06-20T10:00:00Z")

    return meta1, meta2

