import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image

from ..app.schemas.common import Modality, RasterFormat, GeoBBox, ImageMetadata

def detect_format(file_path: str) -> RasterFormat:
    ext = Path(file_path).suffix.lower()
    if ext in [".tif", ".tiff"]:
        return RasterFormat.GEOTIFF
    elif ext == ".png":
        return RasterFormat.PNG
    elif ext in [".jpg", ".jpeg"]:
        return RasterFormat.JPEG
    return RasterFormat.TIFF

def extract_acquisition_date(file_path: str, tags: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extract acquisition date from GeoTIFF tags or filename pattern."""
    if tags:
        date_keys = [
            "TIFFTAG_DATETIME", "ACQUISITION_DATE", "ACQUISITION_DATETIME",
            "DATE", "DATETIME", "SENSING_TIME", "START_TIME", "acquisition_date"
        ]
        for k in date_keys:
            val = tags.get(k) or tags.get(k.lower())
            if val:
                val_str = str(val).strip()
                # Clean up ISO or standard TIFF format YYYY:MM:DD HH:MM:SS
                if re.match(r"^\d{4}:\d{2}:\d{2}", val_str):
                    parts = val_str.split(" ", 1)
                    date_part = parts[0].replace(":", "-")
                    time_part = parts[1] if len(parts) > 1 else "00:00:00"
                    return f"{date_part}T{time_part}"
                return val_str

    # Regex search on filename: e.g. 2023-05-14 or 20230514 or 20230514T120000
    fname = Path(file_path).name
    # Match YYYY-MM-DD or YYYY_MM_DD
    m = re.search(r"((?:19|20)\d{2})[-_](\d{2})[-_](\d{2})", fname)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Match YYYYMMDD
    m2 = re.search(r"((?:19|20)\d{2})(\d{2})(\d{2})", fname)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}"

    return None

def detect_modality(
    file_path: str,
    band_count: int = 3,
    dtype: str = "uint8",
    tags: Optional[Dict[str, Any]] = None,
    value_sample: Optional[np.ndarray] = None,
    explicit_override: Optional[Modality] = None
) -> Tuple[Modality, float]:
    """
    Detects optical/multispectral vs SAR from band count, dtype, value range, metadata tags.
    Returns (Modality, confidence).
    Allows manual override when explicit_override is provided.
    """
    # 1. Manual override
    if explicit_override is not None and explicit_override != Modality.UNKNOWN:
        return explicit_override, 1.0

    fname = Path(file_path).name.lower()
    tags_lower = {str(k).lower(): str(v).lower() for k, v in (tags or {}).items()}

    # 2. Metadata tags inspection
    sar_tag_keywords = ["polarisation", "polarization", "sar", "sentinel-1", "s1", "c-band", "l-band", "radar", "vv", "vh", "hh", "hv"]
    for k, v in tags_lower.items():
        if any(sk in k or sk in v for sk in sar_tag_keywords):
            return Modality.SAR, 0.98

    optical_tag_keywords = ["sentinel-2", "landsat", "msi", "multispectral", "surface_reflectance"]
    for k, v in tags_lower.items():
        if any(ok in k or ok in v for ok in optical_tag_keywords):
            return Modality.MULTISPECTRAL, 0.98

    # 3. Filename patterns
    sar_filename_clues = ["sar", "_vv", "_vh", "_hh", "_hv", "sentinel1", "s1", "cband", "lband", "terrasar", "radarsat"]
    if any(sc in fname for sc in sar_filename_clues):
        return Modality.SAR, 0.92

    optical_filename_clues = ["sentinel2", "s2", "landsat", "planet", "quickbird", "worldview"]
    if any(oc in fname for oc in optical_filename_clues):
        if band_count >= 4:
            return Modality.MULTISPECTRAL, 0.95
        return Modality.OPTICAL, 0.92

    # 4. Band count, dtype, and value range analysis
    if band_count >= 4:
        return Modality.MULTISPECTRAL, 0.90

    # Inspect numerical value range if available
    if value_sample is not None and value_sample.size > 0:
        val_min = float(np.min(value_sample))
        val_max = float(np.max(value_sample))
        val_mean = float(np.mean(value_sample))
        val_std = float(np.std(value_sample))

        # dB SAR values typically have negative values (e.g. -35 dB to +5 dB)
        if val_min < 0 and val_max < 20:
            return Modality.SAR, 0.88

        # Float32 with high speckle coefficient of variation (std/mean > 0.8) and 1-2 bands
        if band_count in (1, 2) and ("float" in dtype.lower() or "int16" in dtype.lower()):
            if val_mean > 0 and (val_std / (val_mean + 1e-6)) > 0.8:
                return Modality.SAR, 0.82

    # 3-band typical RGB optical
    if band_count == 3:
        return Modality.OPTICAL, 0.88

    if band_count in (1, 2):
        if "float" in dtype.lower() or "int16" in dtype.lower():
            return Modality.SAR, 0.70
        return Modality.OPTICAL, 0.75  # panchromatic single-band

    return Modality.OPTICAL, 0.70

def read_geotiff_metadata(
    file_path: str,
    image_id: Optional[str] = None,
    modality_override: Optional[Modality] = None,
    acquisition_date: Optional[str] = None,
    benchmark_mode: bool = False
) -> ImageMetadata:
    """
    Ingests GeoTIFF/TIFF using rasterio. Extracts CRS, transform, bounds, band count,
    dtype, resolution, and acquisition date from tags or filename.
    PNG/JPEG accepted only when benchmark_mode is True.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_fmt = detect_format(file_path)
    img_id = image_id or path.stem

    # Verify format acceptance: PNG/JPEG accepted only when benchmark_mode is enabled
    if file_fmt in [RasterFormat.PNG, RasterFormat.JPEG] and not benchmark_mode:
        raise ValueError(
            f"Format '{file_fmt.value.upper()}' is rejected: standard operation requires GeoTIFF/TIFF. "
            "Set benchmark_mode=True to ingest PNG/JPEG benchmark datasets."
        )

    try:
        with rasterio.open(file_path) as src:
            width = src.width
            height = src.height
            band_count = src.count
            dtype_str = str(src.dtypes[0])
            nodata_val = src.nodata
            crs_str = src.crs.to_string() if src.crs else None
            is_georef = bool(src.crs and src.transform and not src.transform.is_identity)
            transform_list = list(src.transform)[:6] if src.transform else None
            tags = dict(src.tags())

            # Sample data for value range check
            try:
                sample_data = src.read(1, window=rasterio.windows.Window(0, 0, min(64, width), min(64, height)))
            except Exception:
                sample_data = None

            detected_modality, mod_confidence = detect_modality(
                file_path=file_path,
                band_count=band_count,
                dtype=dtype_str,
                tags=tags,
                value_sample=sample_data,
                explicit_override=modality_override
            )

            # Extract acquisition date if not explicitly passed
            final_acq_date = acquisition_date or extract_acquisition_date(file_path, tags)

            geo_bounds: Optional[GeoBBox] = None
            if is_georef:
                left, bottom, right, top = src.bounds
                try:
                    w84_left, w84_bottom, w84_right, w84_top = transform_bounds(
                        src.crs, "EPSG:4326", left, bottom, right, top
                    )
                    geo_bounds = GeoBBox(
                        minx=w84_left,
                        miny=w84_bottom,
                        maxx=w84_right,
                        maxy=w84_top,
                        crs="EPSG:4326"
                    )
                except Exception:
                    geo_bounds = GeoBBox(
                        minx=left,
                        miny=bottom,
                        maxx=right,
                        maxy=top,
                        crs=crs_str or "EPSG:4326"
                    )
            else:
                geo_bounds = GeoBBox(
                    minx=0.0,
                    miny=0.0,
                    maxx=float(width),
                    maxy=float(height),
                    crs="PIXEL"
                )

            res = [abs(src.res[0]), abs(src.res[1])] if is_georef else None

            return ImageMetadata(
                image_id=img_id,
                filename=path.name,
                file_path=str(path.resolve()),
                format=file_fmt,
                modality=detected_modality,
                modality_confidence=round(mod_confidence, 3),
                width=width,
                height=height,
                crs=crs_str,
                bounds=geo_bounds,
                resolution=res,
                transform=transform_list,
                band_count=band_count,
                dtype=dtype_str,
                nodata=nodata_val,
                acquisition_date=final_acq_date,
                is_georeferenced=is_georef,
            )

    except rasterio.errors.RasterioError:
        # Fallback for benchmark images (PNG/JPEG) when benchmark_mode is enabled
        if not benchmark_mode:
            raise ValueError(f"Failed to read raster as GeoTIFF and benchmark_mode is disabled for {file_path}")

        with Image.open(file_path) as img:
            width, height = img.size
            bands = len(img.getbands())
            dtype_str = "uint8"
            detected_modality, mod_confidence = detect_modality(
                file_path=file_path,
                band_count=bands,
                dtype=dtype_str,
                explicit_override=modality_override
            )
            final_acq_date = acquisition_date or extract_acquisition_date(file_path, None)

            return ImageMetadata(
                image_id=img_id,
                filename=path.name,
                file_path=str(path.resolve()),
                format=file_fmt,
                modality=detected_modality,
                modality_confidence=round(mod_confidence, 3),
                width=width,
                height=height,
                crs=None,
                bounds=GeoBBox(minx=0.0, miny=0.0, maxx=float(width), maxy=float(height), crs="PIXEL"),
                resolution=None,
                transform=None,
                band_count=bands,
                dtype=dtype_str,
                nodata=None,
                acquisition_date=final_acq_date,
                is_georeferenced=False,
            )

def read_image_metadata(
    file_path: str,
    image_id: Optional[str] = None,
    modality_hint: Optional[Modality] = None,
    acquisition_date: Optional[str] = None,
    benchmark_mode: bool = False
) -> ImageMetadata:
    """Alias for backwards compatibility and integration."""
    return read_geotiff_metadata(
        file_path=file_path,
        image_id=image_id,
        modality_override=modality_hint,
        acquisition_date=acquisition_date,
        benchmark_mode=benchmark_mode
    )

def read_raster_array(file_path: str) -> np.ndarray:
    """Reads image raster as numpy array (bands, height, width) or (height, width, bands)."""
    try:
        with rasterio.open(file_path) as src:
            return src.read()
    except Exception:
        with Image.open(file_path) as img:
            arr = np.array(img)
            if arr.ndim == 2:
                return arr[np.newaxis, ...]
            elif arr.ndim == 3:
                return np.moveaxis(arr, -1, 0)
            return arr
