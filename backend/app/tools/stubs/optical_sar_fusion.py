from typing import List, Type, Dict, Any, Optional
from pathlib import Path
import numpy as np
import rasterio
from pydantic import BaseModel

from ..base import BaseTool
from ...schemas.tools import OpticalSARFusionInput, OpticalSARFusionOutput
from ...schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from ...ingestion.reader import read_image_metadata, read_raster_array

class OpticalSARFusionStubTool(BaseTool):
    name: str = "optical_sar_fusion"
    description: str = "Two-stream optical and SAR cross-modal segmentation for water bodies and built-up areas."
    input_schema: Type[BaseModel] = OpticalSARFusionInput
    output_schema: Type[BaseModel] = OpticalSARFusionOutput
    whitelisted_params: List[str] = ["optical_path", "sar_path", "target_classes", "confidence_threshold"]

    def run(self, input_data: OpticalSARFusionInput) -> OpticalSARFusionOutput:
        opt_meta = read_image_metadata(input_data.optical_path)
        opt_arr = read_raster_array(input_data.optical_path)
        sar_arr = read_raster_array(input_data.sar_path)

        # Harmonize dimensions
        min_h = min(opt_arr.shape[1], sar_arr.shape[1])
        min_w = min(opt_arr.shape[2], sar_arr.shape[2])

        # Two-stream fusion logic (heuristic/mock rule in Phase 1):
        # In SAR: water has very low backscatter (specular reflection away from sensor).
        # In Optical: water has low red/NIR reflectance.
        sar_slice = sar_arr[0, :min_h, :min_w]
        # Built-up has very high double-bounce SAR backscatter and moderate optical brightness
        sar_norm = (sar_slice - np.min(sar_slice)) / (np.max(sar_slice) - np.min(sar_slice) + 1e-6)

        # Water mask: SAR low (bottom 25%) and Optical moderate/low
        water_mask = (sar_norm < 0.20).astype(np.uint8)
        # Built-up mask: SAR high backscatter (> 0.70)
        built_up_mask = (sar_norm > 0.70).astype(np.uint8)

        total_pixels = float(min_h * min_w)
        water_pixels = int(np.sum(water_mask))
        built_up_pixels = int(np.sum(built_up_mask))

        water_pct = round((water_pixels / total_pixels) * 100.0, 2)
        built_up_pct = round((built_up_pixels / total_pixels) * 100.0, 2)

        # Resolution calculation
        res_m = 10.0
        if opt_meta.resolution and len(opt_meta.resolution) >= 2:
            if opt_meta.is_georeferenced and opt_meta.crs and "4326" in opt_meta.crs:
                res_m = opt_meta.resolution[0] * 111320.0
            else:
                res_m = opt_meta.resolution[0]

        pixel_area_m2 = res_m * res_m
        water_area_km2 = round((water_pixels * pixel_area_m2) / 1_000_000.0, 4)
        built_up_area_km2 = round((built_up_pixels * pixel_area_m2) / 1_000_000.0, 4)

        # Combined classification raster (0: background, 1: water, 2: built-up)
        class_map = np.zeros((min_h, min_w), dtype=np.uint8)
        class_map[water_mask == 1] = 1
        class_map[built_up_mask == 1] = 2

        out_map_path = str(Path("data/outputs") / f"fusion_map_{opt_meta.image_id}.tif")
        Path(out_map_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_map_path,
            'w',
            driver='GTiff',
            height=min_h,
            width=min_w,
            count=1,
            dtype=rasterio.uint8
        ) as dst:
            dst.write(class_map, 1)

        # GeoJSON features for water and built-up bounds
        features = []
        if opt_meta.bounds and opt_meta.is_georeferenced:
            minx, miny, maxx, maxy = opt_meta.bounds.minx, opt_meta.bounds.miny, opt_meta.bounds.maxx, opt_meta.bounds.maxy
            dx = (maxx - minx)
            dy = (maxy - miny)

            # Synthetic sample bounding geometries for visualization
            water_feat = GeoJSONFeature(
                type="Feature",
                geometry=GeoJSONGeometry(
                    type="Polygon",
                    coordinates=[[[minx, miny], [minx + 0.4*dx, miny], [minx + 0.4*dx, miny + 0.4*dy], [minx, miny + 0.4*dy], [minx, miny]]]
                ),
                properties={"class": "water", "area_km2": water_area_km2, "percentage": water_pct}
            )
            built_feat = GeoJSONFeature(
                type="Feature",
                geometry=GeoJSONGeometry(
                    type="Polygon",
                    coordinates=[[[minx + 0.5*dx, miny + 0.5*dy], [maxx, miny + 0.5*dy], [maxx, maxy], [minx + 0.5*dx, maxy], [minx + 0.5*dx, miny + 0.5*dy]]]
                ),
                properties={"class": "built_up", "area_km2": built_up_area_km2, "percentage": built_up_pct}
            )
            features.extend([water_feat, built_feat])

        water_area_m2 = round(water_pixels * pixel_area_m2, 2)
        water_area_ha = round(water_area_m2 / 10_000.0, 4)
        built_up_area_m2 = round(built_up_pixels * pixel_area_m2, 2)
        built_up_area_ha = round(built_up_area_m2 / 10_000.0, 4)

        return OpticalSARFusionOutput(
            classification_map_path=out_map_path,
            water_area_m2=water_area_m2,
            water_area_ha=water_area_ha,
            water_area_km2=water_area_km2,
            water_percentage=water_pct,
            built_up_area_m2=built_up_area_m2,
            built_up_area_ha=built_up_area_ha,
            built_up_area_km2=built_up_area_km2,
            built_up_percentage=built_up_pct,
            geojson=GeoJSONFeatureCollection(type="FeatureCollection", features=features) if features else None,
            ablation_mode=input_data.ablation_mode,
            mode="stub"
        )
