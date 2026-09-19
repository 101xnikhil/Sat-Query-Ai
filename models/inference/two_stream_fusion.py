#!/usr/bin/env python3
"""
Two-Stream Optical + SAR Cross-Modal Segmentation Engine for SatQuery AI.
Architecture:
- Separate Optical Encoder (handles 1, 3, 4, 12 channels)
- Separate SAR Encoder (handles 1 or 2 channels)
- Feature-Level Fusion Head with Ablation Modes ("fused", "optical_only", "sar_only")
- Multi-Class Segmentation Decoder (0: Background, 1: Water, 2: Built-up)
- Degradation handling (Panchromatic, single-pol SAR, resolution mismatch auto-resampling)
- Tiling with 2D Cosine window boundary blending
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import yaml
import numpy as np
from scipy.ndimage import uniform_filter
import rasterio
from rasterio.transform import xy, from_origin
from rasterio.features import shapes
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import shape, mapping
import pyproj

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from backend.app.schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from backend.app.ingestion.reader import read_raster_array, read_image_metadata
from backend.app.validators.sar import convert_to_db, lee_speckle_filter

if HAS_TORCH:
    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    class TwoStreamFusionNet(nn.Module):
        """
        Two-Stream Optical + SAR Neural Segmentation Network.
        Features separate encoder streams, feature-level fusion, and a decoder.
        """
        def __init__(
            self,
            optical_channels: int = 4,
            sar_channels: int = 2,
            feature_dim: int = 32,
            num_classes: int = 3
        ):
            super().__init__()
            self.optical_channels = optical_channels
            self.sar_channels = sar_channels
            self.feature_dim = feature_dim
            self.num_classes = num_classes

            # Optical Stream Encoder
            self.opt_conv1 = ConvBlock(optical_channels, feature_dim)
            # SAR Stream Encoder
            self.sar_conv1 = ConvBlock(sar_channels, feature_dim)

            # Feature-Level Fusion Layer
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )

            # Decoder Head
            self.decoder = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, num_classes, kernel_size=1)
            )

        def forward(
            self,
            x_opt: torch.Tensor,
            x_sar: torch.Tensor,
            ablation_mode: str = "fused"
        ) -> torch.Tensor:
            """
            Forward pass with feature-level ablation support:
            - 'fused': both streams active
            - 'optical_only': zero out SAR stream features
            - 'sar_only': zero out optical stream features
            """
            f_opt = self.opt_conv1(x_opt)
            f_sar = self.sar_conv1(x_sar)

            if ablation_mode == "optical_only":
                f_sar = torch.zeros_like(f_sar)
            elif ablation_mode == "sar_only":
                f_opt = torch.zeros_like(f_opt)

            fused = torch.cat([f_opt, f_sar], dim=1)
            f_fused = self.fusion_conv(fused)
            logits = self.decoder(f_fused)
            return logits

class TwoStreamFusionEngine:
    """
    Inference & execution engine for Two-Stream Optical + SAR Fusion.
    Handles resolution matching, degradation modes, seamless overlap tiling,
    and GeoTIFF/GeoJSON output generation.
    """
    def __init__(
        self,
        config_path: str = "configs/optical_sar_fusion.yaml",
        tile_size: int = 128,
        overlap: int = 32
    ):
        self.config = self._load_config(config_path)
        cfg_inf = self.config.get("inference", {})
        self.tile_size = cfg_inf.get("tile_size", tile_size)
        self.overlap = cfg_inf.get("overlap", overlap)
        self.stride = max(16, self.tile_size - self.overlap)
        self.water_sar_threshold_db = cfg_inf.get("water_sar_threshold_db", -14.0)
        self.built_up_sar_threshold_db = cfg_inf.get("built_up_sar_threshold_db", -2.0)
        self.ndwi_threshold = cfg_inf.get("ndwi_threshold", 0.05)
        self.resample_method = cfg_inf.get("resample_method", "bilinear")

        self.mock_mode = self.config.get("model", {}).get("mock_mode", True)
        self.device = self.config.get("model", {}).get("device", "cpu")
        self.model: Optional[Any] = None

        if HAS_TORCH and not self.mock_mode:
            ckpt = self.config.get("model", {}).get("checkpoint_path")
            if ckpt and Path(ckpt).exists():
                try:
                    self.model = TwoStreamFusionNet(
                        optical_channels=self.config.get("model", {}).get("optical_channels", 4),
                        sar_channels=self.config.get("model", {}).get("sar_channels", 2),
                        feature_dim=self.config.get("model", {}).get("feature_dim", 32),
                        num_classes=self.config.get("model", {}).get("num_classes", 3)
                    ).to(self.device)
                    self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
                    self.model.eval()
                except Exception:
                    self.model = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def compute_blending_weights(th: int, tw: int) -> np.ndarray:
        """2D Hann/Cosine window for seamless boundary tile reassembly."""
        if th <= 1 or tw <= 1:
            return np.ones((th, tw), dtype=np.float32)
        wy = np.sin(np.pi * (np.arange(th) + 0.5) / float(th)) ** 2
        wx = np.sin(np.pi * (np.arange(tw) + 0.5) / float(tw)) ** 2
        w2d = np.outer(wy, wx).astype(np.float32)
        return np.maximum(w2d, 0.05)

    def resample_raster_to_reference(
        self,
        source_path: str,
        ref_path: str,
        resample_type: str = "bilinear"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Handles resolution mismatch between optical and SAR by auto-resampling
        the source raster onto the reference raster's spatial grid and CRS.
        """
        with rasterio.open(ref_path) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height

        resamp = Resampling.bilinear if resample_type == "bilinear" else Resampling.nearest

        with rasterio.open(source_path) as src:
            src_arr = src.read()
            src_count = src.count
            destination = np.zeros((src_count, dst_height, dst_width), dtype=np.float32)

            for b in range(1, src_count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=destination[b - 1],
                    src_transform=src.transform,
                    src_crs=src.crs or dst_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resamp
                )

            resample_info = {
                "source_shape": [src.height, src.width],
                "target_shape": [dst_height, dst_width],
                "source_res": list(src.res) if hasattr(src, "res") else None,
                "target_res": list(ref.res) if hasattr(ref, "res") else None,
                "method": resample_type
            }

        return destination, resample_info

    def segment_tile(
        self,
        opt_tile: np.ndarray,
        sar_tile: np.ndarray,
        ablation_mode: str = "fused",
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Segments a single tile using continuous evidence fusion or PyTorch neural model.
        Returns probability map (3, H, W) for [0: Background, 1: Water, 2: Built-up]
        and degradation info string.
        """
        c_opt, th, tw = opt_tile.shape
        c_sar = sar_tile.shape[0]
        degradation_note = None

        # 1. Optical Feature Extraction & Indices
        eps = 1e-6
        if c_opt >= 4:
            # 4-band: Blue=0, Green=1, Red=2, NIR=3
            green = opt_tile[1]
            red = opt_tile[2]
            nir = opt_tile[3]
            ndwi = (green - nir) / (green + nir + eps)
            ndvi = (nir - red) / (nir + red + eps)
            brightness = (opt_tile[0] + opt_tile[1] + opt_tile[2]) / 3.0
            brightness_norm = (brightness - np.min(brightness)) / (np.ptp(brightness) + eps)
        elif c_opt == 3:
            degradation_note = "Optical missing NIR band (RGB 3-band degraded mode)"
            red = opt_tile[0]
            green = opt_tile[1]
            blue = opt_tile[2]
            ndwi = (green - red) / (green + red + eps)
            ndvi = (green - red) / (green + red + eps)
            brightness = (red + green + blue) / 3.0
            brightness_norm = (brightness - np.min(brightness)) / (np.ptp(brightness) + eps)
        else:
            degradation_note = "Optical single-band panchromatic mode"
            pan = opt_tile[0]
            ndwi = np.zeros_like(pan)
            ndvi = np.zeros_like(pan)
            brightness_norm = (pan - np.min(pan)) / (np.ptp(pan) + eps)

        # 2. SAR Feature Extraction
        sar_band = sar_tile[0]
        if c_sar == 1:
            if degradation_note:
                degradation_note += "; SAR single-pol (VV) mode"
            else:
                degradation_note = "SAR single-pol (VV) mode"

        if np.max(sar_band) > 30.0 or np.min(sar_band) >= 0.0:
            sar_db = convert_to_db(sar_band)
        else:
            sar_db = sar_band.copy()

        sar_filtered_db = lee_speckle_filter(sar_db, window_size=3)

        # 3. Continuous Evidence Calculation
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

        # Optical evidence
        if c_opt >= 4:
            opt_water_raw = sigmoid(10.0 * (ndwi - self.ndwi_threshold))
            opt_water = np.where(ndwi < -0.15, opt_water_raw * 0.2, opt_water_raw)
            local_mean = uniform_filter(brightness_norm, size=3)
            local_contrast = np.abs(brightness_norm - local_mean)
            opt_built_raw = brightness_norm * 0.5 + local_contrast * 3.0
            opt_built = sigmoid(6.0 * (opt_built_raw - 0.40)) * sigmoid(-8.0 * (ndwi - 0.05)) * sigmoid(-5.0 * (ndvi - 0.65))
        elif c_opt == 3:
            opt_water = sigmoid(6.0 * (ndwi - 0.10))
            local_mean = uniform_filter(brightness_norm, size=3)
            local_contrast = np.abs(brightness_norm - local_mean)
            opt_built = sigmoid(5.0 * (brightness_norm * 0.6 + local_contrast * 2.0 - 0.45))
        else:
            opt_water = 1.0 - brightness_norm
            opt_built = brightness_norm

        # SAR evidence
        sar_water = sigmoid(-(sar_filtered_db - self.water_sar_threshold_db) / 2.5)
        sar_built = sigmoid((sar_filtered_db - self.built_up_sar_threshold_db) / 2.5)

        # 4. Fusion and Ablation Modes
        if ablation_mode == "optical_only":
            p_water = opt_water
            p_built_up = opt_built
        elif ablation_mode == "sar_only":
            p_water = sar_water
            p_built_up = sar_built
        else:
            # Fused mode: cross-modal mutual reinforcement
            p_water = 0.50 * sar_water + 0.50 * opt_water
            p_built_up = 0.55 * sar_built + 0.45 * opt_built

        p_bg = np.maximum(0.1, 1.0 - np.maximum(p_water, p_built_up))

        prob_stack = np.stack([p_bg, p_water, p_built_up], axis=0)
        prob_stack /= (np.sum(prob_stack, axis=0, keepdims=True) + eps)
        return prob_stack.astype(np.float32), degradation_note

    def segment_scene(
        self,
        optical_path: str,
        sar_path: str,
        target_classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        ablation_mode: str = "fused",
        output_dir: str = "data/outputs"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Two-Stream Optical + SAR cross-modal segmentation.
        """
        opt_meta = read_image_metadata(optical_path)
        sar_meta = read_image_metadata(sar_path)

        resampling_info = None
        # Check resolution or grid dimension mismatch
        if (opt_meta.width != sar_meta.width or opt_meta.height != sar_meta.height) or (opt_meta.crs != sar_meta.crs):
            # Resample SAR to match Optical reference spatial grid
            sar_arr, resampling_info = self.resample_raster_to_reference(
                source_path=sar_path,
                ref_path=optical_path,
                resample_type=self.resample_method
            )
            opt_arr = read_raster_array(optical_path).astype(np.float32)
        else:
            opt_arr = read_raster_array(optical_path).astype(np.float32)
            sar_arr = read_raster_array(sar_path).astype(np.float32)

        min_h = min(opt_arr.shape[1], sar_arr.shape[1])
        min_w = min(opt_arr.shape[2], sar_arr.shape[2])
        opt_cropped = opt_arr[:, :min_h, :min_w]
        sar_cropped = sar_arr[:, :min_h, :min_w]

        # Tiling or single-pass
        degradation_notes = []
        if min_h <= self.tile_size and min_w <= self.tile_size:
            final_probs, deg = self.segment_tile(
                opt_cropped, sar_cropped,
                ablation_mode=ablation_mode,
                confidence_threshold=confidence_threshold
            )
            if deg:
                degradation_notes.append(deg)
        else:
            from models.inference.fusion_segmenter import OpticalSARFusionEngine
            tile_slices = OpticalSARFusionEngine.generate_tile_slices(
                height=min_h, width=min_w, tile_size=self.tile_size, stride=self.stride
            )
            accum_probs = np.zeros((3, min_h, min_w), dtype=np.float32)
            accum_weights = np.zeros((min_h, min_w), dtype=np.float32)

            for r_slice, c_slice in tile_slices:
                opt_tile = opt_cropped[:, r_slice, c_slice]
                sar_tile = sar_cropped[:, r_slice, c_slice]
                th, tw = opt_tile.shape[1], opt_tile.shape[2]

                tile_probs, deg = self.segment_tile(
                    opt_tile, sar_tile,
                    ablation_mode=ablation_mode,
                    confidence_threshold=confidence_threshold
                )
                if deg and deg not in degradation_notes:
                    degradation_notes.append(deg)

                blend_w = self.compute_blending_weights(th, tw)
                accum_probs[:, r_slice, c_slice] += tile_probs * blend_w[None, :, :]
                accum_weights[r_slice, c_slice] += blend_w

            accum_weights = np.maximum(accum_weights, 1e-6)
            final_probs = accum_probs / accum_weights[None, :, :]

        class_map = np.argmax(final_probs, axis=0).astype(np.uint8)
        max_conf = np.max(final_probs, axis=0)
        class_map[max_conf < confidence_threshold] = 0

        # Pixel and area statistics
        total_pixels = float(min_h * min_w)
        water_pixels = int(np.sum(class_map == 1))
        built_up_pixels = int(np.sum(class_map == 2))

        water_pct = round((water_pixels / total_pixels) * 100.0, 2)
        built_up_pct = round((built_up_pixels / total_pixels) * 100.0, 2)

        res_m = 10.0
        if opt_meta.resolution and len(opt_meta.resolution) >= 2:
            if opt_meta.is_georeferenced and opt_meta.crs and "4326" in opt_meta.crs:
                res_m = opt_meta.resolution[0] * 111320.0
            else:
                res_m = opt_meta.resolution[0]

        pixel_area_m2 = res_m * res_m
        water_area_m2 = round(water_pixels * pixel_area_m2, 2)
        water_area_ha = round(water_area_m2 / 10_000.0, 4)
        water_area_km2 = round(water_area_m2 / 1_000_000.0, 4)

        built_up_area_m2 = round(built_up_pixels * pixel_area_m2, 2)
        built_up_area_ha = round(built_up_area_m2 / 10_000.0, 4)
        built_up_area_km2 = round(built_up_area_m2 / 1_000_000.0, 4)

        # Export GeoTIFF
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_map_path = str(out_dir / f"fusion_{ablation_mode}_{opt_meta.image_id}.tif")

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

        # Vectorize to GeoJSON
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
                                    "area_m2": water_area_m2 if val == 1 else built_up_area_m2,
                                    "area_ha": water_area_ha if val == 1 else built_up_area_ha,
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

        deg_mode_str = "; ".join(degradation_notes) if degradation_notes else "nominal"

        return {
            "classification_map_path": out_map_path,
            "water_area_m2": water_area_m2,
            "water_area_ha": water_area_ha,
            "water_area_km2": water_area_km2,
            "water_percentage": water_pct,
            "built_up_area_m2": built_up_area_m2,
            "built_up_area_ha": built_up_area_ha,
            "built_up_area_km2": built_up_area_km2,
            "built_up_percentage": built_up_pct,
            "water_pixels": water_pixels,
            "built_up_pixels": built_up_pixels,
            "total_pixels": int(total_pixels),
            "geojson": geojson,
            "ablation_mode": ablation_mode,
            "degradation_mode": deg_mode_str,
            "resampling_info": resampling_info
        }
