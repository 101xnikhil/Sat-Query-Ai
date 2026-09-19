from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
import math
import rasterio
from rasterio.transform import xy
import pyproj

from backend.app.schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry

class BoundingBox2D(BaseModel):
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    label: str = "detection"
    confidence: float = 1.0
    normalized: bool = True

def project_boxes_to_geojson(
    boxes: List[BoundingBox2D],
    raster_path: str,
    query: Optional[str] = None
) -> GeoJSONFeatureCollection:
    """
    Projects normalized or pixel bounding boxes to georeferenced WGS84 (EPSG:4326)
    GeoJSON polygon features using rasterio affine transform.
    """
    path = Path(raster_path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    features: List[GeoJSONFeature] = []

    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        is_georef = bool(src.crs and src.transform and not src.transform.is_identity)
        crs = src.crs

        # Prepare CRS transformer if projected CRS != EPSG:4326
        transformer = None
        if is_georef and crs and crs.to_string() != "EPSG:4326":
            try:
                transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            except Exception:
                transformer = None

        for idx, box in enumerate(boxes):
            # 1. Convert to pixel coordinates
            if box.normalized:
                px_min = max(0.0, min(float(width), box.xmin * width))
                px_max = max(0.0, min(float(width), box.xmax * width))
                py_min = max(0.0, min(float(height), box.ymin * height))
                py_max = max(0.0, min(float(height), box.ymax * height))
            else:
                px_min = max(0.0, min(float(width), box.xmin))
                px_max = max(0.0, min(float(width), box.xmax))
                py_min = max(0.0, min(float(height), box.ymin))
                py_max = max(0.0, min(float(height), box.ymax))

            if px_max <= px_min or py_max <= py_min:
                continue

            # 2. Project corner coordinates
            if is_georef:
                # rasterio.transform.xy expects (row, col)
                # 4 corners: TL, TR, BR, BL, TL
                corners_px = [
                    (py_min, px_min),
                    (py_min, px_max),
                    (py_max, px_max),
                    (py_max, px_min),
                    (py_min, px_min),
                ]
                polygon_coords = []
                for row, col in corners_px:
                    x_proj, y_proj = xy(src.transform, row, col)
                    if transformer:
                        lon, lat = transformer.transform(x_proj, y_proj)
                    else:
                        lon, lat = x_proj, y_proj
                    polygon_coords.append([round(lon, 7), round(lat, 7)])
            else:
                # Pixel coordinate grid
                polygon_coords = [
                    [round(px_min, 1), round(py_min, 1)],
                    [round(px_max, 1), round(py_min, 1)],
                    [round(px_max, 1), round(py_max, 1)],
                    [round(px_min, 1), round(py_max, 1)],
                    [round(px_min, 1), round(py_min, 1)],
                ]

            # Calculate area
            box_w_px = px_max - px_min
            box_h_px = py_max - py_min
            pixel_area = box_w_px * box_h_px
            res_x, res_y = src.res if src.res else (10.0, 10.0)
            if is_georef and crs and crs.is_projected:
                area_m2 = round(pixel_area * abs(res_x * res_y), 2)
            elif is_georef and crs:
                # Geographic CRS degrees to approx meters at center latitude
                center_lat = (py_min + py_max) / 2.0
                m_per_deg_lat = 111320.0
                m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat if abs(center_lat) <= 90 else 0))
                area_m2 = round(pixel_area * abs(res_x * m_per_deg_lon * res_y * m_per_deg_lat), 2)
            else:
                # Default 10m Sentinel-2 / Landsat equivalent
                area_m2 = round(pixel_area * 100.0, 2)

            feature = GeoJSONFeature(
                type="Feature",
                geometry=GeoJSONGeometry(type="Polygon", coordinates=[polygon_coords]),
                properties={
                    "id": f"grounding_{idx + 1}",
                    "label": box.label,
                    "confidence": round(box.confidence, 4),
                    "score": round(box.confidence, 4),
                    "pixel_box": [round(px_min, 1), round(py_min, 1), round(px_max, 1), round(py_max, 1)],
                    "area_m2": area_m2,
                    "area_pixels": round(pixel_area, 1),
                    "georeferenced": is_georef,
                    "query": query or ""
                }
            )
            features.append(feature)

    return GeoJSONFeatureCollection(type="FeatureCollection", features=features)
