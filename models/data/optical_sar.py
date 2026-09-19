#!/usr/bin/env python3
"""
Optical-SAR Cross-Modal Dataset Loader for SatQuery AI.
Supports Sen1Floods11 (Water), WHU-OPT-SAR (Water & Built-up), SpaceNet 6 (Built-up/Buildings),
and high-fidelity synthetic benchmark fixtures for reproducible testing.
Includes scale/resolution and SAR speckle noise augmentations.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml
import numpy as np
import rasterio
from rasterio.transform import from_origin

class OpticalSARDataLoader:
    """
    Unified Data Loader for Co-registered Optical + SAR Datasets.
    Supported Benchmarks:
    1. Sen1Floods11: S1 (VV, VH) + S2 (12 bands) at 10m. Hand-labeled water masks. CC BY 4.0.
    2. WHU-OPT-SAR: High-res Optical + SAR pairs with 7-class masks (Water, Built-up). Academic Use.
    3. SpaceNet 6: Capella X-band SAR (0.5m) + WorldView-2 Optical (0.5m) building footprints. CC BY-SA 4.0.
    """
    DATASET_METADATA = {
        "sen1floods11": {
            "name": "Sen1Floods11 (Water & Flood Segmentation)",
            "license": "CC BY 4.0",
            "url": "https://github.com/cloudtostreet/Sen1Floods11",
            "download_cmd": "gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/ ./data/datasets/sen1floods11/"
        },
        "whu_opt_sar": {
            "name": "WHU-OPT-SAR (Multi-class Land Cover Segmentation)",
            "license": "Free for Academic & Non-Commercial Research",
            "url": "https://github.com/AmberHen/WHU-OPT-SAR-dataset",
            "download_cmd": "git clone https://github.com/AmberHen/WHU-OPT-SAR-dataset.git ./data/datasets/whu_opt_sar/"
        },
        "spacenet6": {
            "name": "SpaceNet 6 (Multi-Sensor All-Weather Building Footprints)",
            "license": "CC BY-SA 4.0",
            "url": "https://spacenet.ai/sn6-challenge/",
            "download_cmd": "aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz ./data/datasets/spacenet6/"
        }
    }

    def __init__(
        self,
        dataset_name: str = "sen1floods11",
        data_dir: Optional[str] = None,
        config_path: Optional[str] = "configs/optical_sar_fusion.yaml",
        apply_augmentation: bool = False
    ):
        self.dataset_name = dataset_name.lower()
        self.apply_augmentation = apply_augmentation
        self.samples: List[Dict[str, Any]] = []

        # Resolution and speckle augmentation config
        self.res_jitter_range = (0.5, 2.0)
        self.speckle_looks = 4
        self.speckle_var = 0.15

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                aug_cfg = cfg.get("augmentation", {})
                if aug_cfg.get("resolution_jitter", {}).get("enabled", False):
                    self.res_jitter_range = (
                        aug_cfg["resolution_jitter"].get("min_scale", 0.5),
                        aug_cfg["resolution_jitter"].get("max_scale", 2.0)
                    )
                if aug_cfg.get("speckle_noise", {}).get("enabled", False):
                    self.speckle_looks = aug_cfg["speckle_noise"].get("looks", 4)
                    self.speckle_var = aug_cfg["speckle_noise"].get("variance", 0.15)

        base_dir = Path(data_dir) if data_dir else Path("data/datasets") / self.dataset_name
        self.data_dir = base_dir

        if not self.data_dir.exists():
            self._print_manual_download_instructions()

        self._discover_or_create_samples()

    def _print_manual_download_instructions(self):
        meta = self.DATASET_METADATA.get(self.dataset_name, {
            "name": self.dataset_name,
            "license": "Refer to official source",
            "url": "https://github.com",
            "download_cmd": f"Please place datasets into {self.data_dir}"
        })
        print("=" * 70, file=sys.stderr)
        print(f"📥 OPTICAL-SAR DATASET NOTICE: {meta['name']}", file=sys.stderr)
        print(f"License: {meta['license']}", file=sys.stderr)
        print(f"URL: {meta['url']}", file=sys.stderr)
        print(f"Download command: {meta['download_cmd']}", file=sys.stderr)
        print(f"Target location: {self.data_dir}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    def _discover_or_create_samples(self):
        """Scans for existing GeoTIFF pairs or falls back to synthetic benchmark pairs."""
        if self.data_dir.exists():
            # Check for subdirectories or paired files
            opt_files = sorted(list(self.data_dir.glob("*opt*.tif")) + list(self.data_dir.glob("*S2*.tif")))
            for opt_path in opt_files:
                stem = opt_path.stem.replace("opt", "").replace("S2", "")
                # Find matching SAR file
                sar_candidates = list(self.data_dir.glob(f"*{stem}*sar*.tif")) + list(self.data_dir.glob(f"*{stem}*S1*.tif"))
                mask_candidates = list(self.data_dir.glob(f"*{stem}*mask*.tif")) + list(self.data_dir.glob(f"*{stem}*label*.tif"))
                if sar_candidates and mask_candidates:
                    self.samples.append({
                        "id": stem.strip("_-"),
                        "optical_path": str(opt_path),
                        "sar_path": str(sar_candidates[0]),
                        "mask_path": str(mask_candidates[0])
                    })

        if not self.samples:
            # Generate synthetic benchmark fixture if no real files on disk
            synthetic_dir = Path("data/datasets/synthetic_optical_sar")
            sample = self.create_synthetic_fixture(synthetic_dir, sample_id="bench_pair_01")
            self.samples.append(sample)

    @staticmethod
    def create_synthetic_fixture(
        output_dir: Path,
        sample_id: str = "fixture_01",
        height: int = 128,
        width: int = 128,
        pixel_size_m: float = 10.0
    ) -> Dict[str, Any]:
        """
        Creates a clean co-registered Optical + SAR GeoTIFF pair with known geometric
        ground truth masks (0: Background, 1: Water, 2: Built-up).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        opt_path = output_dir / f"{sample_id}_optical.tif"
        sar_path = output_dir / f"{sample_id}_sar.tif"
        mask_path = output_dir / f"{sample_id}_mask.tif"

        # GeoTransform: centered near Bengaluru (EPSG:4326)
        lon_origin, lat_origin = 77.5946, 12.9716
        deg_res = pixel_size_m / 111320.0
        transform = from_origin(lon_origin, lat_origin, deg_res, deg_res)
        crs = "EPSG:4326"

        # Ground truth mask:
        # Region 1: Water (top-left rectangular reservoir, rows 10:50, cols 10:60)
        # Region 2: Built-up (bottom-right urban settlement, rows 70:120, cols 65:120)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[10:50, 10:60] = 1   # Water: 40 * 50 = 2000 pixels
        mask[70:120, 65:120] = 2 # Built-up: 50 * 55 = 2750 pixels

        # Optical 4-band: Blue (B2), Green (B3), Red (B4), NIR (B8)
        # Background: Moderate reflectance (NDVI ~0.3, NDWI < 0)
        opt_arr = np.zeros((4, height, width), dtype=np.float32)
        opt_arr[0] = 0.12  # Blue
        opt_arr[1] = 0.15  # Green
        opt_arr[2] = 0.14  # Red
        opt_arr[3] = 0.28  # NIR

        # Water in Optical: High Green, Low NIR -> NDWI > 0
        water_sel = (mask == 1)
        opt_arr[0, water_sel] = 0.18
        opt_arr[1, water_sel] = 0.25
        opt_arr[2, water_sel] = 0.08
        opt_arr[3, water_sel] = 0.02

        # Built-up in Optical: High visible, low NIR absorption, high texture
        built_sel = (mask == 2)
        opt_arr[0, built_sel] = 0.35
        opt_arr[1, built_sel] = 0.36
        opt_arr[2, built_sel] = 0.38
        opt_arr[3, built_sel] = 0.35

        # Add slight natural Gaussian noise
        opt_arr += np.random.normal(0, 0.01, opt_arr.shape).astype(np.float32)
        opt_arr = np.clip(opt_arr, 0.0, 1.0)

        # SAR Dual-pol: Band 0 = VV, Band 1 = VH
        # Background: Moderate vegetation backscatter (-12 dB VV, -18 dB VH)
        sar_arr = np.zeros((2, height, width), dtype=np.float32)
        sar_arr[0] = -12.0
        sar_arr[1] = -18.0

        # Water in SAR: Smooth specular reflection -> very low backscatter (<= -18 dB VV, -25 dB VH)
        sar_arr[0, water_sel] = -21.0
        sar_arr[1, water_sel] = -27.0

        # Built-up in SAR: Double-bounce corner reflection -> strong backscatter (>= 0 dB VV, -5 dB VH)
        sar_arr[0, built_sel] = 2.5
        sar_arr[1, built_sel] = -3.0

        # Add speckle noise to SAR
        sar_arr += np.random.normal(0, 0.8, sar_arr.shape).astype(np.float32)

        # Write Optical GeoTIFF
        with rasterio.open(
            opt_path, 'w',
            driver='GTiff',
            height=height, width=width,
            count=4, dtype=rasterio.float32,
            crs=crs, transform=transform
        ) as dst:
            dst.write(opt_arr)

        # Write SAR GeoTIFF
        with rasterio.open(
            sar_path, 'w',
            driver='GTiff',
            height=height, width=width,
            count=2, dtype=rasterio.float32,
            crs=crs, transform=transform
        ) as dst:
            dst.write(sar_arr)

        # Write Mask GeoTIFF
        with rasterio.open(
            mask_path, 'w',
            driver='GTiff',
            height=height, width=width,
            count=1, dtype=rasterio.uint8,
            crs=crs, transform=transform
        ) as dst:
            dst.write(mask, 1)

        return {
            "id": sample_id,
            "optical_path": str(opt_path),
            "sar_path": str(sar_path),
            "mask_path": str(mask_path),
            "height": height,
            "width": width,
            "pixel_size_m": pixel_size_m,
            "water_pixels": int(np.sum(mask == 1)),
            "built_up_pixels": int(np.sum(mask == 2)),
            "total_pixels": int(height * width)
        }

    def apply_speckle_noise(self, sar_arr: np.ndarray) -> np.ndarray:
        """
        Applies multiplicative Gamma speckle noise modeling SAR radar speckle.
        """
        shape = sar_arr.shape
        gamma_noise = np.random.gamma(shape=self.speckle_looks, scale=1.0 / self.speckle_looks, size=shape)
        # Apply to linear power, then return in dB
        # Assuming sar_arr is in dB:
        sar_linear = 10.0 ** (sar_arr / 10.0)
        speckled_linear = sar_linear * gamma_noise
        speckled_db = 10.0 * np.log10(np.maximum(speckled_linear, 1e-6))
        return speckled_db.astype(np.float32)

    def apply_resolution_jitter(self, arr: np.ndarray) -> np.ndarray:
        """
        Simulates resolution / scale degradation by downsampling then upsampling.
        """
        scale = np.random.uniform(self.res_jitter_range[0], self.res_jitter_range[1])
        c, h, w = arr.shape
        new_h = max(8, int(h * scale))
        new_w = max(8, int(w * scale))
        # Simple nearest-neighbor/bilinear simulation using indexing
        y_indices = np.linspace(0, h - 1, new_h).astype(int)
        x_indices = np.linspace(0, w - 1, new_w).astype(int)
        resampled = arr[:, y_indices[:, None], x_indices]
        # Resample back to original h, w
        y_back = np.linspace(0, new_h - 1, h).astype(int)
        x_back = np.linspace(0, new_w - 1, w).astype(int)
        restored = resampled[:, y_back[:, None], x_back]
        return restored.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        with rasterio.open(item["optical_path"]) as src:
            opt = src.read().astype(np.float32)
        with rasterio.open(item["sar_path"]) as src:
            sar = src.read().astype(np.float32)
        with rasterio.open(item["mask_path"]) as src:
            mask = src.read(1).astype(np.int64)

        if self.apply_augmentation:
            sar = self.apply_speckle_noise(sar)
            opt = self.apply_resolution_jitter(opt)

        return {
            "id": item["id"],
            "optical": opt,
            "sar": sar,
            "mask": mask,
            "optical_path": item["optical_path"],
            "sar_path": item["sar_path"]
        }
