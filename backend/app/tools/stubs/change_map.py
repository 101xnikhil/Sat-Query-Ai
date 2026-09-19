from typing import List, Type, Dict, Any, Optional
from pathlib import Path
import numpy as np
import rasterio
from pydantic import BaseModel

from ..base import BaseTool
from ...schemas.tools import ChangeMapInput, ChangeMapOutput
from ...schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from ...ingestion.reader import read_image_metadata, read_raster_array

class ChangeMapStubTool(BaseTool):
    name: str = "change_map"
    description: str = "Generates pixel-level change mask and calculates empirical area changed and direction."
    input_schema: Type[BaseModel] = ChangeMapInput
    output_schema: Type[BaseModel] = ChangeMapOutput
    whitelisted_params: List[str] = ["image_t1_path", "image_t2_path", "threshold", "min_area_m2"]

    def run(self, input_data: ChangeMapInput) -> ChangeMapOutput:
        meta1 = read_image_metadata(input_data.image_t1_path)
        meta2 = read_image_metadata(input_data.image_t2_path)
        
        arr1 = read_raster_array(input_data.image_t1_path).astype(np.float32)
        arr2 = read_raster_array(input_data.image_t2_path).astype(np.float32)

        # Match dimensions if needed
        min_h = min(arr1.shape[1], arr2.shape[1])
        min_w = min(arr1.shape[2], arr2.shape[2])
        diff = np.abs(arr1[:, :min_h, :min_w] - arr2[:, :min_h, :min_w])
        diff_mean = np.mean(diff, axis=0)

        # Normalize difference to 0..1
        max_val = np.max(diff_mean) if np.max(diff_mean) > 0 else 1.0
        norm_diff = diff_mean / max_val
        
        # Binary change mask based on threshold
        change_mask = (norm_diff > input_data.threshold).astype(np.uint8)

        total_pixels = change_mask.size
        changed_pixels = int(np.sum(change_mask))
        percentage_changed = round((changed_pixels / total_pixels) * 100.0, 2)

        # Estimate ground resolution (default 10m Sentinel-2 / 30m Landsat if unknown)
        res_m = 10.0
        if meta1.resolution and len(meta1.resolution) >= 2:
            # If in degrees, convert approx 1 deg ~ 111,000m
            if meta1.is_georeferenced and meta1.crs and "4326" in meta1.crs:
                res_m = meta1.resolution[0] * 111320.0
            else:
                res_m = meta1.resolution[0]
        
        pixel_area_m2 = res_m * res_m
        area_changed_m2 = changed_pixels * pixel_area_m2
        area_changed_km2 = round(area_changed_m2 / 1_000_000.0, 4)

        # Determine dominant direction of change from mean intensity shift
        intensity_shift = np.mean(arr2[:, :min_h, :min_w]) - np.mean(arr1[:, :min_h, :min_w])
        direction = "construction / surface brightness increase" if intensity_shift > 0 else "vegetation decrease / darkening"

        # Save change mask raster
        out_mask_path = str(Path("data/outputs") / f"change_mask_{meta1.image_id}_{meta2.image_id}.tif")
        Path(out_mask_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save simple GeoTIFF
        with rasterio.open(
            out_mask_path,
            'w',
            driver='GTiff',
            height=min_h,
            width=min_w,
            count=1,
            dtype=rasterio.uint8
        ) as dst:
            dst.write(change_mask, 1)

        # Create bounding GeoJSON feature of changed zone
        geojson = None
        if meta1.bounds and meta1.is_georeferenced:
            minx, miny, maxx, maxy = meta1.bounds.minx, meta1.bounds.miny, meta1.bounds.maxx, meta1.bounds.maxy
            feature = GeoJSONFeature(
                type="Feature",
                geometry=GeoJSONGeometry(
                    type="Polygon",
                    coordinates=[[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]
                ),
                properties={
                    "area_changed_km2": area_changed_km2,
                    "percentage_changed": percentage_changed,
                    "direction": direction
                }
            )
            geojson = GeoJSONFeatureCollection(type="FeatureCollection", features=[feature])

        return ChangeMapOutput(
            change_mask_path=out_mask_path,
            area_changed_km2=area_changed_km2,
            percentage_changed=percentage_changed,
            direction_of_change=direction,
            geojson=geojson,
            mode="stub"
        )
