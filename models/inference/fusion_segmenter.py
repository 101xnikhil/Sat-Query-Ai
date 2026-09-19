import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.ndimage import uniform_filter
import rasterio
from rasterio.transform import xy
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import pyproj

from backend.app.schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from backend.app.ingestion.reader import read_raster_array, read_image_metadata
from backend.app.validators.sar import convert_to_db, lee_speckle_filter

class OpticalSARFusionEngine:
    """
    Two-Stream Optical + SAR Cross-Modal Segmentation Engine with Overlap Tiling
    and Seamless Blending Reassembly.
    
    Physical Principles:
    - SAR Radar: Calm water exhibits specular reflection away from sensor (very low backscatter <= -15 dB).
                 Built-up structures exhibit cardinal wall corner / double-bounce reflections (>= 0 dB).
    - Optical: Multi-spectral bands provide spectral absorption/reflectance signatures
               (NDWI = (Green - NIR)/(Green + NIR), NDVI = (NIR - Red)/(NIR + Red)).
    """
    def __init__(
        self,
        tile_size: int = 128,
        overlap: int = 32,
        water_sar_threshold_db: float = -14.0,
        built_up_sar_threshold_db: float = -2.0,
        ndwi_threshold: float = 0.05
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = max(16, tile_size - overlap)
        self.water_sar_threshold_db = water_sar_threshold_db
        self.built_up_sar_threshold_db = built_up_sar_threshold_db
        self.ndwi_threshold = ndwi_threshold

    @staticmethod
    def generate_tile_slices(height: int, width: int, tile_size: int = 128, stride: int = 96) -> List[Tuple[slice, slice]]:
        """Generates grid of overlapping tile slices covering the entire raster."""
        r_slices = []
        r = 0
        while r + tile_size < height:
            r_slices.append(slice(r, r + tile_size))
            r += stride
        r_slices.append(slice(max(0, height - tile_size), height))

        c_slices = []
        c = 0
        while c + tile_size < width:
            c_slices.append(slice(c, c + tile_size))
            c += stride
        c_slices.append(slice(max(0, width - tile_size), width))

        # Unique slices preserving order
        unique_r = []
        for s in r_slices:
            if s not in unique_r:
                unique_r.append(s)

        unique_c = []
        for s in c_slices:
            if s not in unique_c:
                unique_c.append(s)

        tile_grid = []
        for rs in unique_r:
            for cs in unique_c:
                tile_grid.append((rs, cs))
        return tile_grid

    @staticmethod
    def compute_blending_weights(th: int, tw: int) -> np.ndarray:
        """
        Computes 2D smooth cosine/linear weighting window for seamless boundary blending.
        Eliminates seam artifacts where tiles overlap.
        """
        if th <= 1 or tw <= 1:
            return np.ones((th, tw), dtype=np.float32)

        # 1D Hann/Cosine window with positive baseline
        wy = np.sin(np.pi * (np.arange(th) + 0.5) / float(th)) ** 2
        wx = np.sin(np.pi * (np.arange(tw) + 0.5) / float(tw)) ** 2
        w2d = np.outer(wy, wx).astype(np.float32)
        # Avoid pure zero at boundary edges so single-coverage pixels still retain weight
        w2d = np.maximum(w2d, 0.05)
        return w2d

    def segment_tile(
        self,
        opt_tile: np.ndarray,
        sar_tile: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Performs cross-modal two-stream segmentation on a single tile.
        Returns probability map of shape (3, H, W) for [0: background, 1: water, 2: built_up].
        """
        c_opt, th, tw = opt_tile.shape
        c_sar = sar_tile.shape[0]

        # 1. Optical Feature Extraction & Indices
        eps = 1e-6
        if c_opt >= 4:
            # Standard 4-band: Blue=0, Green=1, Red=2, NIR=3
            green = opt_tile[1]
            red = opt_tile[2]
            nir = opt_tile[3]
            ndwi = (green - nir) / (green + nir + eps)
            ndvi = (nir - red) / (nir + red + eps)
            brightness = (opt_tile[0] + opt_tile[1] + opt_tile[2]) / 3.0
            brightness_norm = (brightness - np.min(brightness)) / (np.ptp(brightness) + eps)
        elif c_opt == 3:
            # RGB fallback: Red=0, Green=1, Blue=2
            red = opt_tile[0]
            green = opt_tile[1]
            blue = opt_tile[2]
            ndwi = (green - red) / (green + red + eps)
            ndvi = (green - red) / (green + red + eps)
            brightness = (red + green + blue) / 3.0
            brightness_norm = (brightness - np.min(brightness)) / (np.ptp(brightness) + eps)
        else:
            # Single band panchromatic
            pan = opt_tile[0]
            ndwi = np.zeros_like(pan)
            ndvi = np.zeros_like(pan)
            brightness_norm = (pan - np.min(pan)) / (np.ptp(pan) + eps)

        # 2. SAR Feature Extraction (dB conversion & speckle reduction)
        sar_band = sar_tile[0]
        # Check if already in dB (typically negative or moderate values) or linear amplitude
        if np.max(sar_band) > 30.0 or np.min(sar_band) >= 0.0:
            sar_db = convert_to_db(sar_band)
        else:
            sar_db = sar_band.copy()

        # Apply Lee speckle filter on SAR dB
        sar_filtered_db = lee_speckle_filter(sar_db, window_size=3)

        # Normalize SAR dB range [-25 dB, +10 dB] into [0, 1]
        sar_norm = np.clip((sar_filtered_db - (-25.0)) / (10.0 - (-25.0)), 0.0, 1.0)

        # 3. Two-Stream Continuous Evidence Fusion
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

        # Water Evidence:
        # In SAR: Specular reflection away from sensor -> low backscatter (<= -14 dB)
        sar_water_evidence = sigmoid(-(sar_filtered_db - self.water_sar_threshold_db) / 2.5)

        if c_opt >= 3:
            # In Optical: NDWI > 0.0
            opt_water_evidence = sigmoid(10.0 * (ndwi - self.ndwi_threshold))
            p_water = 0.50 * sar_water_evidence + 0.50 * opt_water_evidence
            # Cross-modal suppression: if NDWI is strongly negative (vegetation/desert), suppress water
            p_water = np.where(ndwi < -0.15, p_water * 0.2, p_water)
        else:
            # Panchromatic degradation
            p_water = 0.70 * sar_water_evidence + 0.30 * (1.0 - brightness_norm)

        # Built-up Evidence:
        # In SAR: Intense double-bounce corner reflection (>= -2 dB)
        sar_built_evidence = sigmoid((sar_filtered_db - self.built_up_sar_threshold_db) / 2.5)

        # In Optical: High visible reflectance, high texture contrast, not water
        local_mean = uniform_filter(brightness_norm, size=3)
        local_contrast = np.abs(brightness_norm - local_mean)
        opt_built_raw = brightness_norm * 0.5 + local_contrast * 3.0
        opt_built_evidence = sigmoid(6.0 * (opt_built_raw - 0.40))

        if c_opt >= 3:
            # Suppress built-up if NDWI is high (water) or NDVI is very high (dense forest)
            opt_built_evidence = opt_built_evidence * sigmoid(-8.0 * (ndwi - 0.05)) * sigmoid(-5.0 * (ndvi - 0.65))

        p_built_up = 0.55 * sar_built_evidence + 0.45 * opt_built_evidence

        # Background / Natural Vegetation:
        p_bg = np.maximum(0.1, 1.0 - np.maximum(p_water, p_built_up))

        # Stack into 3-channel probability volume
        prob_stack = np.stack([p_bg, p_water, p_built_up], axis=0)
        sum_prob = np.sum(prob_stack, axis=0, keepdims=True) + eps
        prob_stack /= sum_prob
        return prob_stack.astype(np.float32)

    def segment_scene(
        self,
        optical_path: str,
        sar_path: str,
        target_classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        output_dir: str = "data/outputs"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end two-stream Optical+SAR segmentation:
        1. Reads and harmonizes rasters.
        2. Tiles large scenes with 25% overlap window.
        3. Fuses multi-modal probabilities with 2D cosine blending reassembly.
        4. Derives empirical surface areas (km²) and percentages.
        5. Exports georeferenced GeoTIFF and vectorized GeoJSON polygon layers.
        """
        opt_meta = read_image_metadata(optical_path)
        sar_meta = read_image_metadata(sar_path)

        opt_arr = read_raster_array(optical_path).astype(np.float32)
        sar_arr = read_raster_array(sar_path).astype(np.float32)

        # Common spatial extent
        min_h = min(opt_arr.shape[1], sar_arr.shape[1])
        min_w = min(opt_arr.shape[2], sar_arr.shape[2])

        opt_cropped = opt_arr[:, :min_h, :min_w]
        sar_cropped = sar_arr[:, :min_h, :min_w]

        # Check if tiling needed or process as single tile
        if min_h <= self.tile_size and min_w <= self.tile_size:
            # Single tile execution
            tile_prob = self.segment_tile(opt_cropped, sar_cropped, confidence_threshold)
            final_probs = tile_prob
        else:
            # Overlap Tiling & Merging
            tile_slices = self.generate_tile_slices(
                height=min_h,
                width=min_w,
                tile_size=self.tile_size,
                stride=self.stride
            )

            accum_probs = np.zeros((3, min_h, min_w), dtype=np.float32)
            accum_weights = np.zeros((min_h, min_w), dtype=np.float32)

            for r_slice, c_slice in tile_slices:
                opt_tile = opt_cropped[:, r_slice, c_slice]
                sar_tile = sar_cropped[:, r_slice, c_slice]
                th = opt_tile.shape[1]
                tw = opt_tile.shape[2]

                tile_probs = self.segment_tile(opt_tile, sar_tile, confidence_threshold)
                blend_w = self.compute_blending_weights(th, tw)

                accum_probs[:, r_slice, c_slice] += tile_probs * blend_w[None, :, :]
                accum_weights[r_slice, c_slice] += blend_w

            accum_weights = np.maximum(accum_weights, 1e-6)
            final_probs = accum_probs / accum_weights[None, :, :]

        # Discrete classification map: 0 = background, 1 = water, 2 = built_up
        class_map = np.argmax(final_probs, axis=0).astype(np.uint8)

        # Apply confidence threshold mask: if confidence < threshold, set to background (0)
        max_conf = np.max(final_probs, axis=0)
        class_map[max_conf < confidence_threshold] = 0

        # Exact pixel counts and empirical spatial metrics
        total_pixels = float(min_h * min_w)
        water_pixels = int(np.sum(class_map == 1))
        built_up_pixels = int(np.sum(class_map == 2))

        water_pct = round((water_pixels / total_pixels) * 100.0, 2)
        built_up_pct = round((built_up_pixels / total_pixels) * 100.0, 2)

        # Ground resolution in meters
        res_m = 10.0
        if opt_meta.resolution and len(opt_meta.resolution) >= 2:
            if opt_meta.is_georeferenced and opt_meta.crs and "4326" in opt_meta.crs:
                res_m = opt_meta.resolution[0] * 111320.0
            else:
                res_m = opt_meta.resolution[0]

        pixel_area_m2 = res_m * res_m
        water_area_km2 = round((water_pixels * pixel_area_m2) / 1_000_000.0, 4)
        built_up_area_km2 = round((built_up_pixels * pixel_area_m2) / 1_000_000.0, 4)

        # Export georeferenced classification GeoTIFF
        out_path_dir = Path(output_dir)
        out_path_dir.mkdir(parents=True, exist_ok=True)
        out_map_path = str(out_path_dir / f"fusion_map_{opt_meta.image_id}.tif")

        with rasterio.open(optical_path) as src:
            profile = src.profile.copy()
            profile.update({
                'driver': 'GTiff',
                'height': min_h,
                'width': min_w,
                'count': 1,
                'dtype': rasterio.uint8,
                'nodata': 0
            })

        with rasterio.open(out_map_path, 'w', **profile) as dst:
            dst.write(class_map, 1)

        # Vectorize water and built-up zones to GeoJSON
        geojson_features: List[GeoJSONFeature] = []
        if (water_pixels > 0 or built_up_pixels > 0) and opt_meta.is_georeferenced:
            with rasterio.open(out_map_path) as mask_src:
                geom_shapes = shapes(class_map, mask=class_map > 0, transform=mask_src.transform)
                transformer = None
                if mask_src.crs and mask_src.crs.to_string() != "EPSG:4326":
                    try:
                        transformer = pyproj.Transformer.from_crs(mask_src.crs, "EPSG:4326", always_xy=True)
                    except Exception:
                        transformer = None

                idx = 0
                for geom, val in geom_shapes:
                    if val in (1, 2):
                        poly = shape(geom)
                        if poly.is_valid and poly.area > 0:
                            poly_geojson = mapping(poly)
                            if transformer:
                                new_coords = []
                                for ring in poly_geojson["coordinates"]:
                                    new_ring = [list(transformer.transform(x, y)) for x, y in ring]
                                    new_coords.append(new_ring)
                                poly_geojson["coordinates"] = new_coords

                            class_name = "water" if val == 1 else "built_up"
                            feat = GeoJSONFeature(
                                type="Feature",
                                geometry=GeoJSONGeometry(type=poly_geojson["type"], coordinates=poly_geojson["coordinates"]),
                                properties={
                                    "id": f"fusion_{class_name}_{idx + 1}",
                                    "class": class_name,
                                    "area_km2": water_area_km2 if val == 1 else built_up_area_km2,
                                    "percentage": water_pct if val == 1 else built_up_pct,
                                    "confidence": round(float(np.mean(max_conf[class_map == val])), 4)
                                }
                            )
                            geojson_features.append(feat)
                            idx += 1
                            if len(geojson_features) >= 60:
                                break

        geojson = GeoJSONFeatureCollection(type="FeatureCollection", features=geojson_features) if geojson_features else None

        return {
            "classification_map_path": out_map_path,
            "water_area_km2": water_area_km2,
            "built_up_area_km2": built_up_area_km2,
            "water_percentage": water_pct,
            "built_up_percentage": built_up_pct,
            "water_pixels": water_pixels,
            "built_up_pixels": built_up_pixels,
            "total_pixels": int(total_pixels),
            "geojson": geojson
        }
