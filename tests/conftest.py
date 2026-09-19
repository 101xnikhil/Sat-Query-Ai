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
