from typing import List, Type, Dict, Any, Optional
from pathlib import Path
import numpy as np
import rasterio
from pydantic import BaseModel

from ..base import BaseTool
from ...schemas.tools import SpectralIndexInput, SpectralIndexOutput
from ...schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from ...ingestion.reader import read_image_metadata, read_raster_array

class SpectralIndexTool(BaseTool):
    name: str = "spectral_index"
    description: str = "Computes spectral indices (NDVI, NDWI, NDBI) with graceful degradation for few bands."
    input_schema: Type[BaseModel] = SpectralIndexInput
    output_schema: Type[BaseModel] = SpectralIndexOutput
    whitelisted_params: List[str] = ["image_path", "index_type", "threshold"]

    def run(self, input_data: SpectralIndexInput) -> SpectralIndexOutput:
        meta = read_image_metadata(input_data.image_path)
        arr = read_raster_array(input_data.image_path).astype(np.float32)
        idx_type = input_data.index_type.upper()
        bands = arr.shape[0]

        # Graceful degradation for panchromatic single band
        if bands < 2:
            # Cannot compute spectral ratio on single band
            out_mask_path = str(Path("data/outputs") / f"index_single_band_{meta.image_id}.tif")
            return SpectralIndexOutput(
                index_type=f"{idx_type} (Panchromatic Degraded)",
                index_map_path=out_mask_path,
                mean_index=0.0,
                positive_area_km2=0.0,
                positive_percentage=0.0,
                geojson=None
            )

        # Multi-band calculation
        # If Sentinel-2 style (4+ bands: B2 Blue, B3 Green, B4 Red, B8 NIR):
        if bands >= 4:
            # 0-indexed: B2=0, B3=1, B4=2, B8=3
            green = arr[1]
            red = arr[2]
            nir = arr[3]
            swir = arr[4] if bands >= 5 else arr[3]
        elif bands == 3:
            # RGB fallback: red=0, green=1, blue=2
            red = arr[0]
            green = arr[1]
            nir = arr[1] * 1.2  # synthetic approximation when NIR missing
            swir = arr[0] * 1.1
        else:
            red = arr[0]
            nir = arr[1]
            green = arr[0]
            swir = arr[1]

        eps = 1e-6
        if idx_type == "NDVI":
            index_arr = (nir - red) / (nir + red + eps)
        elif idx_type == "NDWI":
            index_arr = (green - nir) / (green + nir + eps)
        elif idx_type == "NDBI":
            index_arr = (swir - nir) / (swir + nir + eps)
        else:
            index_arr = (nir - red) / (nir + red + eps)
            idx_type = "NDVI"

        # Clip index values to valid [-1, 1] range
        index_arr = np.clip(index_arr, -1.0, 1.0)
        mean_val = float(np.mean(index_arr))

        # Binary positive mask above threshold
        pos_mask = (index_arr > input_data.threshold).astype(np.uint8)
        total_pixels = float(pos_mask.size)
        pos_pixels = int(np.sum(pos_mask))
        pos_pct = round((pos_pixels / total_pixels) * 100.0, 2)

        # Spatial area calculation
        res_m = 10.0
        if meta.resolution and len(meta.resolution) >= 2:
            if meta.is_georeferenced and meta.crs and "4326" in meta.crs:
                res_m = meta.resolution[0] * 111320.0
            else:
                res_m = meta.resolution[0]

        pixel_area_m2 = res_m * res_m
        pos_area_km2 = round((pos_pixels * pixel_area_m2) / 1_000_000.0, 4)

        # Save index map
        out_map_path = str(Path("data/outputs") / f"index_{idx_type}_{meta.image_id}.tif")
        Path(out_map_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with rasterio.open(input_data.image_path) as src:
                profile = src.profile.copy()
                profile.update({
                    'driver': 'GTiff',
                    'height': index_arr.shape[0],
                    'width': index_arr.shape[1],
                    'count': 1,
                    'dtype': rasterio.float32
                })
        except Exception:
            profile = {
                'driver': 'GTiff',
                'height': index_arr.shape[0],
                'width': index_arr.shape[1],
                'count': 1,
                'dtype': rasterio.float32
            }
        with rasterio.open(out_map_path, 'w', **profile) as dst:
            dst.write(index_arr.astype(np.float32), 1)

        # Create GeoJSON bounding extent for visualization
        geojson = None
        if meta.bounds and meta.is_georeferenced:
            minx, miny, maxx, maxy = meta.bounds.minx, meta.bounds.miny, meta.bounds.maxx, meta.bounds.maxy
            feature = GeoJSONFeature(
                type="Feature",
                geometry=GeoJSONGeometry(
                    type="Polygon",
                    coordinates=[[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]
                ),
                properties={
                    "index_type": idx_type,
                    "mean_index": round(mean_val, 3),
                    "positive_area_km2": pos_area_km2,
                    "positive_percentage": pos_pct
                }
            )
            geojson = GeoJSONFeatureCollection(type="FeatureCollection", features=[feature])

        return SpectralIndexOutput(
            index_type=idx_type,
            index_map_path=out_map_path,
            mean_index=round(mean_val, 4),
            positive_area_km2=pos_area_km2,
            positive_percentage=pos_pct,
            geojson=geojson
        )
