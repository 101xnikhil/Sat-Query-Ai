import os
import shutil
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

@pytest.fixture(scope="session")
def test_data_dir():
    d = tempfile.mkdtemp(prefix="satquery_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture(scope="session")
def optical_geotiff(test_data_dir):
    """Generates a 4-band (B, G, R, NIR) GeoTIFF in EPSG:4326 over a 0.05 x 0.05 deg extent."""
    path = os.path.join(test_data_dir, "optical_sample.tif")
    width, height = 100, 100
    # Geographic bounds: around 77.0 to 77.05 E, 28.0 to 28.05 N (Delhi area)
    minx, miny, maxx, maxy = 77.0, 28.0, 77.05, 28.05
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Synthetic bands:
    # Band 1: Blue, Band 2: Green, Band 3: Red, Band 4: NIR
    np.random.seed(42)
    b1 = np.random.randint(20, 80, (height, width), dtype=np.uint8)
    b2 = np.random.randint(30, 100, (height, width), dtype=np.uint8)
    b3 = np.random.randint(40, 120, (height, width), dtype=np.uint8)
    b4 = np.random.randint(100, 240, (height, width), dtype=np.uint8)  # High NIR
    data = np.stack([b1, b2, b3, b4], axis=0)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=4,
        dtype=rasterio.uint8,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(data)

    return path

@pytest.fixture(scope="session")
def sar_geotiff(test_data_dir):
    """Generates a 1-band SAR amplitude GeoTIFF in EPSG:4326 matching optical bounds."""
    path = os.path.join(test_data_dir, "sar_sample_vv.tif")
    width, height = 100, 100
    minx, miny, maxx, maxy = 77.0, 28.0, 77.05, 28.05
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    np.random.seed(99)
    # Rayleigh / Exponential-like SAR speckle amplitude
    sar_data = np.random.gamma(shape=2.0, scale=15.0, size=(1, height, width)).astype(np.float32)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(sar_data)

    return path

@pytest.fixture(scope="session")
def utm_reproject_geotiff(test_data_dir):
    """Generates a GeoTIFF in UTM zone 43N (EPSG:32643) covering the same Delhi region."""
    path = os.path.join(test_data_dir, "utm_sample.tif")
    width, height = 100, 100
    # Approx UTM 43N coords for 77.0 E, 28.0 N
    minx, miny, maxx, maxy = 696000.0, 3099000.0, 701000.0, 3104000.0
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    data = np.random.randint(50, 200, (3, height, width), dtype=np.uint8)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=rasterio.uint8,
        crs='EPSG:32643',
        transform=transform
    ) as dst:
        dst.write(data)

    return path

@pytest.fixture(scope="session")
def disjoint_geotiff(test_data_dir):
    """Generates a GeoTIFF completely geographically disjoint (e.g. 10.0 deg away)."""
    path = os.path.join(test_data_dir, "disjoint_sample.tif")
    width, height = 50, 50
    minx, miny, maxx, maxy = 12.0, 45.0, 12.05, 45.05  # Europe
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    data = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=rasterio.uint8,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(data)

    return path

@pytest.fixture(scope="session")
def benchmark_png(test_data_dir):
    """Generates a standard benchmark PNG image without georeferencing."""
    path = os.path.join(test_data_dir, "benchmark_img.png")
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
    return path
