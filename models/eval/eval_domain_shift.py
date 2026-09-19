#!/usr/bin/env python3
"""
Domain-Shift Robustness Evaluation Script for SatQuery AI.
Simulates a high-resolution domain shift (e.g., simulating Cartosat-2S / RISAT vs Sentinel-1/2)
by scale-shifting and resampling the validation set, then reporting the empirical metric drop.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import rasterio

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.data.optical_sar import OpticalSARDataLoader
from models.inference.two_stream_fusion import TwoStreamFusionEngine
from models.eval.eval_fusion import compute_class_metrics

def run_domain_shift_test(
    dataset_name: str = "sen1floods11",
    shift_factor: float = 2.0,
    config_path: str = "configs/optical_sar_fusion.yaml",
    output_path: str = "data/outputs/domain_shift_eval.json"
) -> Dict[str, Any]:
    """
    Executes baseline vs shifted domain evaluation.
    shift_factor > 1.0 simulates higher-resolution / resolution scale shift.
    """
    print("=" * 70)
    print("🛰️  SatQuery AI - Domain-Shift Robustness Evaluation")
    print(f"Dataset: {dataset_name} | Shift Factor: {shift_factor}x | Target Domain: Cartosat-2S / RISAT")
    print("=" * 70)

    loader = OpticalSARDataLoader(dataset_name=dataset_name, config_path=config_path)
    engine = TwoStreamFusionEngine(config_path=config_path)

    baseline_water_ious = []
    baseline_built_ious = []
    shifted_water_ious = []
    shifted_built_ious = []

    shift_dir = Path("data/outputs/domain_shift_temp")
    shift_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(loader)):
        sample = loader[i]
        opt_path = sample["optical_path"]
        sar_path = sample["sar_path"]
        target_mask = sample["mask"]

        # 1. Baseline Run
        base_res = engine.segment_scene(
            optical_path=opt_path,
            sar_path=sar_path,
            ablation_mode="fused",
            output_dir="data/outputs"
        )
        with rasterio.open(base_res["classification_map_path"]) as msrc:
            base_pred = msrc.read(1)

        min_h = min(base_pred.shape[0], target_mask.shape[0])
        min_w = min(base_pred.shape[1], target_mask.shape[1])
        b_w = compute_class_metrics(base_pred[:min_h, :min_w], target_mask[:min_h, :min_w], class_val=1)
        b_b = compute_class_metrics(base_pred[:min_h, :min_w], target_mask[:min_h, :min_w], class_val=2)
        baseline_water_ious.append(b_w["iou"])
        baseline_built_ious.append(b_b["iou"])

        # 2. Resampled / Domain-Shifted Run (simulate Cartosat/RISAT high-resolution sampling shift)
        with rasterio.open(opt_path) as src:
            opt_data = src.read()
            opt_prof = src.profile.copy()

        with rasterio.open(sar_path) as src:
            sar_data = src.read()
            sar_prof = src.profile.copy()

        # Resample resolution grid by shift_factor
        new_h = int(opt_data.shape[1] * shift_factor)
        new_w = int(opt_data.shape[2] * shift_factor)

        # Bilinear interpolation via scipy/numpy
        y_idx = np.linspace(0, opt_data.shape[1] - 1, new_h).astype(int)
        x_idx = np.linspace(0, opt_data.shape[2] - 1, new_w).astype(int)
        shifted_opt = opt_data[:, y_idx[:, None], x_idx]
        shifted_sar = sar_data[:, y_idx[:, None], x_idx]

        shifted_opt_path = shift_dir / f"shifted_opt_{sample['id']}.tif"
        shifted_sar_path = shift_dir / f"shifted_sar_{sample['id']}.tif"

        opt_prof.update({"height": new_h, "width": new_w})
        sar_prof.update({"height": new_h, "width": new_w})

        with rasterio.open(shifted_opt_path, "w", **opt_prof) as dst:
            dst.write(shifted_opt)
        with rasterio.open(shifted_sar_path, "w", **sar_prof) as dst:
            dst.write(shifted_sar)

        # Run inference on shifted resolution
        shift_res = engine.segment_scene(
            optical_path=str(shifted_opt_path),
            sar_path=str(shifted_sar_path),
            ablation_mode="fused",
            output_dir=str(shift_dir)
        )
        with rasterio.open(shift_res["classification_map_path"]) as msrc:
            shift_pred = msrc.read(1)

        # Downsample prediction back to target grid to evaluate
        y_back = np.linspace(0, new_h - 1, min_h).astype(int)
        x_back = np.linspace(0, new_w - 1, min_w).astype(int)
        shift_pred_orig_grid = shift_pred[y_back[:, None], x_back]

        s_w = compute_class_metrics(shift_pred_orig_grid, target_mask[:min_h, :min_w], class_val=1)
        s_b = compute_class_metrics(shift_pred_orig_grid, target_mask[:min_h, :min_w], class_val=2)
        shifted_water_ious.append(s_w["iou"])
        shifted_built_ious.append(s_b["iou"])

    base_w_mean = round(float(np.mean(baseline_water_ious)), 4)
    base_b_mean = round(float(np.mean(baseline_built_ious)), 4)
    base_miou = round(float((base_w_mean + base_b_mean) / 2.0), 4)

    shift_w_mean = round(float(np.mean(shifted_water_ious)), 4)
    shift_b_mean = round(float(np.mean(shifted_built_ious)), 4)
    shift_miou = round(float((shift_w_mean + shift_b_mean) / 2.0), 4)

    delta_miou = round(base_miou - shift_miou, 4)
    pct_drop = round((delta_miou / (base_miou + 1e-6)) * 100.0, 2)

    report = {
        "dataset": dataset_name,
        "shift_factor": shift_factor,
        "target_domain": "Cartosat-2S (0.6m) / RISAT-1A (C-band)",
        "baseline": {
            "water_iou": base_w_mean,
            "built_up_iou": base_b_mean,
            "mean_class_iou": base_miou
        },
        "domain_shifted": {
            "water_iou": shift_w_mean,
            "built_up_iou": shift_b_mean,
            "mean_class_iou": shift_miou
        },
        "metric_drop": {
            "delta_mean_class_iou": delta_miou,
            "relative_drop_pct": pct_drop
        }
    }

    print("\n📈 DOMAIN-SHIFT SIMULATION RESULTS:")
    print("-" * 70)
    print(f"Baseline Domain (Sentinel 10m):         mIoU = {base_miou:.4f} (Water: {base_w_mean:.4f}, Built-up: {base_b_mean:.4f})")
    print(f"Shifted Domain (Cartosat/RISAT Sim):     mIoU = {shift_miou:.4f} (Water: {shift_w_mean:.4f}, Built-up: {shift_b_mean:.4f})")
    print(f"Observed Metric Drop:                   ΔmIoU = -{delta_miou:.4f} ({pct_drop}% drop)")
    print("-" * 70)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved domain shift report to: {out_p}\n")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Domain-Shift Robustness")
    parser.add_argument("--dataset", default="sen1floods11")
    parser.add_argument("--shift-factor", type=float, default=2.0)
    parser.add_argument("--config", default="configs/optical_sar_fusion.yaml")
    parser.add_argument("--output", default="data/outputs/domain_shift_eval.json")
    args = parser.parse_args()

    run_domain_shift_test(
        dataset_name=args.dataset,
        shift_factor=args.shift_factor,
        config_path=args.config,
        output_path=args.output
    )
