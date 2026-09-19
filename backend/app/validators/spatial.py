import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_geom
from shapely.geometry import box, shape
from shapely.ops import transform
import pyproj

from ..schemas.common import ImageMetadata, GeoBBox

def compute_extent_overlap(meta1: ImageMetadata, meta2: ImageMetadata) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Computes spatial overlap IoU and intersection area between two images.
    If images are georeferenced, uses EPSG:4326 bounding boxes.
    If pixel-based, compares normalized or pixel extents.
    """
    if not meta1.bounds or not meta2.bounds:
        return 0.0, None

    # Both georeferenced in EPSG:4326
    if meta1.is_georeferenced and meta2.is_georeferenced:
        box1 = box(meta1.bounds.minx, meta1.bounds.miny, meta1.bounds.maxx, meta1.bounds.maxy)
        box2 = box(meta2.bounds.minx, meta2.bounds.miny, meta2.bounds.maxx, meta2.bounds.maxy)

        if not box1.intersects(box2):
            return 0.0, {"intersection_area": 0.0, "box1": box1.bounds, "box2": box2.bounds}

        intersection = box1.intersection(box2).area
        union = box1.union(box2).area
        iou = intersection / union if union > 0 else 0.0

        return iou, {
            "intersection_area": intersection,
            "union_area": union,
            "iou": round(iou, 4)
        }
    else:
        # Non-georeferenced benchmark images (assume same coordinate grid if dimensions match)
        if meta1.width == meta2.width and meta1.height == meta2.height:
            return 1.0, {"mode": "pixel_grid", "iou": 1.0}
        
        # Approximate overlap if dimensions differ
        min_w = min(meta1.width, meta2.width)
        min_h = min(meta1.height, meta2.height)
        max_w = max(meta1.width, meta2.width)
        max_h = max(meta1.height, meta2.height)
        iou = (min_w * min_h) / (max_w * max_h)
        return round(iou, 4), {"mode": "pixel_dimensions", "iou": round(iou, 4)}

def reproject_raster_to_match(
    source_path: str,
    reference_path: str,
    output_path: str,
    resampling_method: Resampling = Resampling.bilinear
) -> Dict[str, Any]:
    """
    Reprojects source raster to match CRS, resolution, and extent of reference raster.
    """
    with rasterio.open(reference_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    with rasterio.open(source_path) as src:
        src_crs = src.crs
        if not src_crs:
            src_crs = "EPSG:4326"

        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'driver': 'GTiff'
        })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )

    return {
        "source": source_path,
        "reference": reference_path,
        "output": output_path,
        "src_crs": str(src_crs),
        "dst_crs": str(dst_crs),
        "method": resampling_method.name,
    }
