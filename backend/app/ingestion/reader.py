# Re-exporting from unified backend/ingest/reader.py
from ..schemas.common import Modality, RasterFormat, GeoBBox, ImageMetadata
from ...ingest.reader import (
    detect_format,
    detect_modality,
    extract_acquisition_date,
    read_geotiff_metadata,
    read_image_metadata,
    read_raster_array
)

def inspect_modality(file_path: str, band_count: int, explicit_modality=None):
    mod, _ = detect_modality(file_path, band_count=band_count, explicit_override=explicit_modality)
    return mod

__all__ = [
    "detect_format",
    "detect_modality",
    "inspect_modality",
    "extract_acquisition_date",
    "read_geotiff_metadata",
    "read_image_metadata",
    "read_raster_array",
    "Modality",
    "RasterFormat",
    "GeoBBox",
    "ImageMetadata"
]
