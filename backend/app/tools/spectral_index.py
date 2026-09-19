import os
from pathlib import Path
from typing import List, Type, Dict, Any, Optional
import numpy as np
import rasterio
import yaml
from pydantic import BaseModel

from .base import BaseTool
from ..schemas.tools import SpectralIndexInput, SpectralIndexOutput
from ..schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from ..ingestion.reader import read_image_metadata, read_raster_array

class SpectralIndexTool(BaseTool):
    """
    Real Spectral Index Tool for SatQuery AI.
    Calculates NDWI, NDVI, NDBI strictly based on configured sensor band mappings.
    If required bands are missing (e.g. RGB image lacks NIR, or Panchromatic 1-band),
    returns status='not_applicable' cleanly without fabricating synthetic data.
    """
    name: str = "spectral_index"
    description: str = "Computes spectral indices (NDVI, NDWI, NDBI) strictly requiring respective bands or returns not_applicable."
    input_schema: Type[BaseModel] = SpectralIndexInput
    output_schema: Type[BaseModel] = SpectralIndexOutput
    whitelisted_params: List[str] = ["image_path", "index_type", "threshold", "sensor_profile"]

    def __init__(self, config_path: str = "configs/sensor_profiles.yaml"):
        super().__init__()
        self.sensor_profiles = self._load_sensor_profiles(config_path)

    def _load_sensor_profiles(self, path_str: str) -> Dict[str, Any]:
        p = Path(path_str)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get("sensors", {})
        return {}

    def run(self, input_data: SpectralIndexInput) -> SpectralIndexOutput:
        meta = read_image_metadata(input_data.image_path)
        arr = read_raster_array(input_data.image_path).astype(np.float32)
        idx_type = input_data.index_type.upper().strip()
        bands = arr.shape[0]

        # Determine band indices from sensor profile or band count
        sensor = input_data.sensor_profile or "sentinel2"
        profile = self.sensor_profiles.get(sensor, {})
        prof_bands = profile.get("bands", {})

        green_idx: Optional[int] = None
        red_idx: Optional[int] = None
        nir_idx: Optional[int] = None
        swir_idx: Optional[int] = None

        if prof_bands:
            green_idx = prof_bands.get("green")
            red_idx = prof_bands.get("red")
            nir_idx = prof_bands.get("nir")
            swir_idx = prof_bands.get("swir1") or prof_bands.get("swir")
        else:
            # Heuristic default based on standard Sentinel-2 / Landsat order
            if bands >= 5:
                green_idx, red_idx, nir_idx, swir_idx = 1, 2, 3, 4
            elif bands == 4:
                green_idx, red_idx, nir_idx, swir_idx = 1, 2, 3, None
            elif bands == 3:
                red_idx, green_idx, nir_idx, swir_idx = 0, 1, None, None

        # Check band validity against actual raster bands count
        def valid_idx(idx: Optional[int]) -> bool:
            return idx is not None and 0 <= idx < bands

        green_valid = valid_idx(green_idx)
        red_valid = valid_idx(red_idx)
        nir_valid = valid_idx(nir_idx)
        swir_valid = valid_idx(swir_idx)

        # Enforce strict band presence per requested index
        if idx_type == "NDWI":
            # NDWI = (Green - NIR) / (Green + NIR)
            if not green_valid or not nir_valid:
                reason = f"NDWI requires Green and NIR bands. Image has {bands} band(s); NIR band is absent."
                return SpectralIndexOutput(
                    index_type=idx_type,
                    status="not_applicable",
                    reason=reason,
                    index_map_path=None,
                    mean_index=None,
                    positive_area_m2=0.0,
                    positive_area_ha=0.0,
                    positive_area_km2=0.0,
                    positive_percentage=0.0,
                    geojson=None,
                    mode="model"
                )
            b_high = arr[green_idx]
            b_low = arr[nir_idx]

        elif idx_type == "NDVI":
            # NDVI = (NIR - Red) / (NIR + Red)
            if not red_valid or not nir_valid:
                reason = f"NDVI requires Red and NIR bands. Image has {bands} band(s); NIR band is absent."
                return SpectralIndexOutput(
                    index_type=idx_type,
                    status="not_applicable",
                    reason=reason,
                    index_map_path=None,
                    mean_index=None,
                    positive_area_m2=0.0,
                    positive_area_ha=0.0,
                    positive_area_km2=0.0,
                    positive_percentage=0.0,
                    geojson=None,
                    mode="model"
                )
            b_high = arr[nir_idx]
            b_low = arr[red_idx]

        elif idx_type == "NDBI":
            # NDBI = (SWIR - NIR) / (SWIR + NIR)
            if not swir_valid or not nir_valid:
                reason = f"NDBI requires SWIR and NIR bands. Image has {bands} band(s); SWIR/NIR band is absent."
                return SpectralIndexOutput(
                    index_type=idx_type,
                    status="not_applicable",
                    reason=reason,
                    index_map_path=None,
                    mean_index=None,
                    positive_area_m2=0.0,
                    positive_area_ha=0.0,
                    positive_area_km2=0.0,
                    positive_percentage=0.0,
                    geojson=None,
                    mode="model"
                )
            b_high = arr[swir_idx]
            b_low = arr[nir_idx]

        else:
            return SpectralIndexOutput(
                index_type=idx_type,
                status="not_applicable",
                reason=f"Unknown or unsupported spectral index '{idx_type}'. Supported: NDWI, NDVI, NDBI.",
                index_map_path=None,
                mean_index=None,
                positive_area_m2=0.0,
                positive_area_ha=0.0,
                positive_area_km2=0.0,
                positive_percentage=0.0,
                geojson=None,
                mode="model"
            )

        # Computation
        eps = 1e-6
        index_arr = (b_high - b_low) / (b_high + b_low + eps)
        index_arr = np.clip(index_arr, -1.0, 1.0)
        mean_val = float(np.mean(index_arr))

        # Positive detection mask
        pos_mask = (index_arr > input_data.threshold).astype(np.uint8)
        total_pixels = float(pos_mask.size)
        pos_pixels = int(np.sum(pos_mask))
        pos_pct = round((pos_pixels / total_pixels) * 100.0, 2)

        # Ground resolution in meters
        res_m = 10.0
        if meta.resolution and len(meta.resolution) >= 2:
            if meta.is_georeferenced and meta.crs and "4326" in meta.crs:
                res_m = meta.resolution[0] * 111320.0
            else:
                res_m = meta.resolution[0]

        pixel_area_m2 = res_m * res_m
        pos_area_m2 = round(pos_pixels * pixel_area_m2, 2)
        pos_area_ha = round(pos_area_m2 / 10_000.0, 4)
        pos_area_km2 = round(pos_area_m2 / 1_000_000.0, 4)

        # Export GeoTIFF
        out_dir = Path("data/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_map_path = str(out_dir / f"index_{idx_type}_{meta.image_id}.tif")

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

        # GeoJSON extent
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
                    "positive_area_m2": pos_area_m2,
                    "positive_area_ha": pos_area_ha,
                    "positive_area_km2": pos_area_km2,
                    "positive_percentage": pos_pct
                }
            )
            geojson = GeoJSONFeatureCollection(type="FeatureCollection", features=[feature])

        return SpectralIndexOutput(
            index_type=idx_type,
            status="computed",
            reason=None,
            index_map_path=out_map_path,
            mean_index=round(mean_val, 4),
            positive_area_m2=pos_area_m2,
            positive_area_ha=pos_area_ha,
            positive_area_km2=pos_area_km2,
            positive_percentage=pos_pct,
            geojson=geojson,
            mode="model"
        )
