import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image

from ..schemas.common import Modality, RasterFormat, GeoBBox, ImageMetadata

def detect_format(file_path: str) -> RasterFormat:
    ext = Path(file_path).suffix.lower()
    if ext in [".tif", ".tiff"]:
        return RasterFormat.GEOTIFF
    elif ext == ".png":
        return RasterFormat.PNG
    elif ext in [".jpg", ".jpeg"]:
        return RasterFormat.JPEG
    return RasterFormat.TIFF

def inspect_modality(file_path: str, band_count: int, explicit_modality: Optional[Modality] = None) -> Modality:
    if explicit_modality and explicit_modality != Modality.UNKNOWN:
        return explicit_modality
    
    fname = Path(file_path).name.lower()
    if any(k in fname for k in ["sar", "_vv", "_vh", "_hh", "_hv", "sentinel1", "s1"]):
        return Modality.SAR
    if band_count >= 4 or "sentinel2" in fname or "s2" in fname or "landsat" in fname:
        return Modality.MULTISPECTRAL
    return Modality.OPTICAL

def read_image_metadata(
    file_path: str,
    image_id: Optional[str] = None,
    modality_hint: Optional[Modality] = None,
    acquisition_date: Optional[str] = None
) -> ImageMetadata:
    """Extract comprehensive raster metadata, CRS, bounding box and resolution."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_fmt = detect_format(file_path)
    img_id = image_id or path.stem

    # Attempt rasterio read for geospatial / multi-band data
    try:
        with rasterio.open(file_path) as src:
            width = src.width
            height = src.height
            band_count = src.count
            dtype_str = str(src.dtypes[0])
            nodata_val = src.nodata
            crs_str = src.crs.to_string() if src.crs else None
            is_georef = bool(src.crs and src.transform and not src.transform.is_identity)
            
            geo_bounds: Optional[GeoBBox] = None
            if is_georef:
                left, bottom, right, top = src.bounds
                try:
                    # Convert to WGS84 EPSG:4326 for unified geo-queries
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
                # Fallback to pixel coordinate bounding box
                geo_bounds = GeoBBox(
                    minx=0.0,
                    miny=0.0,
                    maxx=float(width),
                    maxy=float(height),
                    crs="PIXEL"
                )

            res = [abs(src.res[0]), abs(src.res[1])] if is_georef else None
            modality = inspect_modality(file_path, band_count, modality_hint)

            return ImageMetadata(
                image_id=img_id,
                filename=path.name,
                file_path=str(path.resolve()),
                format=file_fmt,
                modality=modality,
                width=width,
                height=height,
                crs=crs_str,
                bounds=geo_bounds,
                resolution=res,
                band_count=band_count,
                dtype=dtype_str,
                nodata=nodata_val,
                acquisition_date=acquisition_date,
                is_georeferenced=is_georef,
            )
    except rasterio.errors.RasterioError:
        # Fallback to PIL for non-geotiff standard images
        with Image.open(file_path) as img:
            width, height = img.size
            bands = len(img.getbands())
            dtype_str = "uint8"
            modality = inspect_modality(file_path, bands, modality_hint)

            return ImageMetadata(
                image_id=img_id,
                filename=path.name,
                file_path=str(path.resolve()),
                format=file_fmt,
                modality=modality,
                width=width,
                height=height,
                crs=None,
                bounds=GeoBBox(minx=0.0, miny=0.0, maxx=float(width), maxy=float(height), crs="PIXEL"),
                resolution=None,
                band_count=bands,
                dtype=dtype_str,
                nodata=None,
                acquisition_date=acquisition_date,
                is_georeferenced=False,
            )

def read_raster_array(file_path: str) -> np.ndarray:
    """Reads image raster as numpy array (bands, height, width) or (height, width, bands)."""
    try:
        with rasterio.open(file_path) as src:
            data = src.read()
            return data
    except Exception:
        with Image.open(file_path) as img:
            arr = np.array(img)
            if arr.ndim == 2:
                return arr[np.newaxis, ...]  # (1, H, W)
            elif arr.ndim == 3:
                return np.moveaxis(arr, -1, 0)  # (C, H, W)
            return arr
