import pytest
from backend.app.ingestion.reader import read_image_metadata, read_raster_array, detect_format
from backend.app.schemas.common import Modality, RasterFormat

def test_read_geotiff_metadata(optical_geotiff):
    meta = read_image_metadata(optical_geotiff)
    assert meta.format == RasterFormat.GEOTIFF
    assert meta.width == 100
    assert meta.height == 100
    assert meta.band_count == 4
    assert meta.is_georeferenced is True
    assert meta.bounds is not None
    assert meta.bounds.minx == pytest.approx(77.0, rel=1e-3)
    assert meta.bounds.maxx == pytest.approx(77.05, rel=1e-3)
    assert meta.modality in [Modality.MULTISPECTRAL, Modality.OPTICAL]

def test_read_sar_metadata(sar_geotiff):
    meta = read_image_metadata(sar_geotiff)
    assert meta.format == RasterFormat.GEOTIFF
    assert meta.modality == Modality.SAR
    assert meta.band_count == 1

def test_read_png_metadata(benchmark_png):
    meta = read_image_metadata(benchmark_png)
    assert meta.format == RasterFormat.PNG
    assert meta.width == 128
    assert meta.height == 128
    assert meta.band_count == 3
    assert meta.is_georeferenced is False
    assert meta.bounds.crs == "PIXEL"

def test_read_raster_array(optical_geotiff):
    arr = read_raster_array(optical_geotiff)
    assert arr.shape == (4, 100, 100)
