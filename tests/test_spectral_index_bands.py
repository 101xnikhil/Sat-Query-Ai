import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.app.tools.spectral_index import SpectralIndexTool
from backend.app.schemas.tools import SpectralIndexInput

@pytest.fixture
def multi_band_geotiff(tmp_path):
    """Creates a 4-band GeoTIFF (Blue, Green, Red, NIR)."""
    p = tmp_path / "s2_4band.tif"
    h, w = 40, 40
    arr = np.zeros((4, h, w), dtype=np.float32)
    arr[0] = 0.10  # Blue
    arr[1] = 0.25  # Green
    arr[2] = 0.12  # Red
    arr[3] = 0.05  # NIR (low NIR -> high NDWI water)
    
    transform = from_origin(77.59, 12.97, 0.0001, 0.0001)
    with rasterio.open(
        p, "w", driver="GTiff", height=h, width=w, count=4,
        dtype=rasterio.float32, crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(arr)
    return str(p)

@pytest.fixture
def rgb_3band_geotiff(tmp_path):
    """Creates a 3-band RGB GeoTIFF (Red, Green, Blue) without NIR."""
    p = tmp_path / "rgb_3band.tif"
    h, w = 40, 40
    arr = np.random.uniform(0.1, 0.8, (3, h, w)).astype(np.float32)
    transform = from_origin(77.59, 12.97, 0.0001, 0.0001)
    with rasterio.open(
        p, "w", driver="GTiff", height=h, width=w, count=3,
        dtype=rasterio.float32, crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(arr)
    return str(p)

@pytest.fixture
def pan_1band_geotiff(tmp_path):
    """Creates a single-band Panchromatic GeoTIFF."""
    p = tmp_path / "pan_1band.tif"
    h, w = 40, 40
    arr = np.random.uniform(0.1, 0.9, (1, h, w)).astype(np.float32)
    transform = from_origin(77.59, 12.97, 0.0001, 0.0001)
    with rasterio.open(
        p, "w", driver="GTiff", height=h, width=w, count=1,
        dtype=rasterio.float32, crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(arr)
    return str(p)

def test_spectral_index_4band_ndwi_computed(multi_band_geotiff):
    """NDWI should be successfully computed when Green and NIR exist."""
    tool = SpectralIndexTool()
    res = tool.run(SpectralIndexInput(
        image_path=multi_band_geotiff,
        index_type="NDWI",
        threshold=0.0
    ))
    assert res.status == "computed"
    assert res.mean_index is not None
    assert res.mean_index > 0.5  # (0.25 - 0.05)/(0.25 + 0.05) ~ 0.66
    assert res.index_map_path is not None
    assert os.path.exists(res.index_map_path)
    assert res.positive_percentage > 90.0

def test_spectral_index_rgb_missing_nir_returns_not_applicable(rgb_3band_geotiff):
    """RGB lacking NIR must return not_applicable cleanly for NDWI."""
    tool = SpectralIndexTool()
    res = tool.run(SpectralIndexInput(
        image_path=rgb_3band_geotiff,
        index_type="NDWI",
        threshold=0.0
    ))
    assert res.status == "not_applicable"
    assert res.mean_index is None
    assert res.index_map_path is None
    assert "NIR" in res.reason
    assert res.positive_area_km2 == 0.0

def test_spectral_index_rgb_missing_nir_returns_not_applicable_ndvi(rgb_3band_geotiff):
    """RGB lacking NIR must return not_applicable cleanly for NDVI."""
    tool = SpectralIndexTool()
    res = tool.run(SpectralIndexInput(
        image_path=rgb_3band_geotiff,
        index_type="NDVI",
        threshold=0.2
    ))
    assert res.status == "not_applicable"
    assert res.mean_index is None
    assert "NIR" in res.reason

def test_spectral_index_panchromatic_single_band_degradation(pan_1band_geotiff):
    """Single band panchromatic returns not_applicable without crashing."""
    tool = SpectralIndexTool()
    res = tool.run(SpectralIndexInput(
        image_path=pan_1band_geotiff,
        index_type="NDWI",
        threshold=0.0
    ))
    assert res.status == "not_applicable"
    assert "band" in res.reason
    assert res.mean_index is None
