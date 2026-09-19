#!/usr/bin/env python3
"""
Two-Stream Optical + SAR Fusion Evaluation Script for SatQuery AI.
Computes per-class Intersection-over-Union (IoU) and F1 scores for water and built-up classes.
Supports full ablation analysis (optical_only, sar_only, fused) with real empirical measurements.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.data.optical_sar import OpticalSARDataLoader
from models.inference.two_stream_fusion import TwoStreamFusionEngine

def compute_class_metrics(pred_mask: np.ndarray, target_mask: np.ndarray, class_val: int) -> Dict[str, float]:
    """Computes empirical IoU, Precision, Recall, and F1 for a specific class."""
    eps = 1e-6
    pred_c = (pred_mask == class_val)
    target_c = (target_mask == class_val)

    intersection = np.sum(pred_c & target_c)
    union = np.sum(pred_c | target_c)
    pred_sum = np.sum(pred_c)
    target_sum = np.sum(target_c)

    iou = float((intersection + eps) / (union + eps)) if union > 0 else 1.0
    precision = float((intersection + eps) / (pred_sum + eps)) if pred_sum > 0 else 0.0
    recall = float((intersection + eps) / (target_sum + eps)) if target_sum > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall + eps)) if (precision + recall) > 0 else 0.0

    return {
        "iou": round(iou, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "target_pixels": int(target_sum),
        "pred_pixels": int(pred_sum)
    }

def run_evaluation(
    dataset_name: str = "sen1floods11",
    ablation_mode: str = "all",
    config_path: str = "configs/optical_sar_fusion.yaml",
    output_path: str = "data/outputs/fusion_ablation_eval.json"
) -> Dict[str, Any]:
    """
    Evaluates Optical + SAR fusion across ablation modes:
    - 'fused': Dual-stream cross-modal segmentation
    - 'optical_only': Optical stream alone
    - 'sar_only': SAR stream alone
    """
    print("=" * 70)
    print("🛰️  SatQuery AI - Optical + SAR Two-Stream Fusion Ablation Evaluation")
    print(f"Dataset: {dataset_name} | Ablation: {ablation_mode} | Config: {config_path}")
    print("=" * 70)

    loader = OpticalSARDataLoader(dataset_name=dataset_name, config_path=config_path)
    engine = TwoStreamFusionEngine(config_path=config_path)

    modes_to_test = ["optical_only", "sar_only", "fused"] if ablation_mode == "all" else [ablation_mode]
    results: Dict[str, Any] = {"dataset": dataset_name, "samples_evaluated": len(loader), "modes": {}}

    for mode in modes_to_test:
        mode_water_ious = []
        mode_water_f1s = []
        mode_built_ious = []
        mode_built_f1s = []

        for i in range(len(loader)):
            sample = loader[i]
            opt_path = sample["optical_path"]
            sar_path = sample["sar_path"]
            target_mask = sample["mask"]

            res = engine.segment_scene(
                optical_path=opt_path,
                sar_path=sar_path,
                ablation_mode=mode,
                output_dir="data/outputs"
            )

            # Read generated mask
            with open(res["classification_map_path"], "rb"):
                import rasterio
                with rasterio.open(res["classification_map_path"]) as msrc:
                    pred_mask = msrc.read(1)

            # Match sizes
            min_h = min(pred_mask.shape[0], target_mask.shape[0])
            min_w = min(pred_mask.shape[1], target_mask.shape[1])
            p_crop = pred_mask[:min_h, :min_w]
            t_crop = target_mask[:min_h, :min_w]

            w_metrics = compute_class_metrics(p_crop, t_crop, class_val=1)
            b_metrics = compute_class_metrics(p_crop, t_crop, class_val=2)

            mode_water_ious.append(w_metrics["iou"])
            mode_water_f1s.append(w_metrics["f1"])
            mode_built_ious.append(b_metrics["iou"])
            mode_built_f1s.append(b_metrics["f1"])

        mean_w_iou = round(float(np.mean(mode_water_ious)), 4)
        mean_w_f1 = round(float(np.mean(mode_water_f1s)), 4)
        mean_b_iou = round(float(np.mean(mode_built_ious)), 4)
        mean_b_f1 = round(float(np.mean(mode_built_f1s)), 4)
        mean_m_iou = round(float((mean_w_iou + mean_b_iou) / 2.0), 4)

        results["modes"][mode] = {
            "water": {"mean_iou": mean_w_iou, "mean_f1": mean_w_f1},
            "built_up": {"mean_iou": mean_b_iou, "mean_f1": mean_b_f1},
            "mean_class_iou": mean_m_iou
        }

    # Print clean comparison table
    print("\n📊 ABLATION BENCHMARK RESULTS (Real Runs):")
    print("-" * 70)
    print(f"{'Ablation Mode':<18} | {'Water IoU':<10} | {'Water F1':<10} | {'Built-up IoU':<13} | {'Built-up F1':<12} | {'mIoU':<8}")
    print("-" * 70)
    for mode, data in results["modes"].items():
        w_iou = data["water"]["mean_iou"]
        w_f1 = data["water"]["mean_f1"]
        b_iou = data["built_up"]["mean_iou"]
        b_f1 = data["built_up"]["mean_f1"]
        m_iou = data["mean_class_iou"]
        print(f"{mode:<18} | {w_iou:<10.4f} | {w_f1:<10.4f} | {b_iou:<13.4f} | {b_f1:<12.4f} | {m_iou:<8.4f}")
    print("-" * 70)

    # Calculate fusion gain
    if "fused" in results["modes"] and "optical_only" in results["modes"]:
        gain_opt = results["modes"]["fused"]["mean_class_iou"] - results["modes"]["optical_only"]["mean_class_iou"]
        gain_sar = results["modes"]["fused"]["mean_class_iou"] - results["modes"]["sar_only"]["mean_class_iou"]
        results["fusion_gain_over_optical"] = round(gain_opt, 4)
        results["fusion_gain_over_sar"] = round(gain_sar, 4)
        print(f"✨ Fusion Benefit: +{gain_opt*100:.2f}% mIoU over Optical-only, +{gain_sar*100:.2f}% mIoU over SAR-only")

    # Save to JSON
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical evaluation to: {out_p}\n")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Optical + SAR Fusion Model across Ablation Modes")
    parser.add_argument("--dataset", default="sen1floods11")
    parser.add_argument("--ablation", default="all", choices=["all", "optical_only", "sar_only", "fused"])
    parser.add_argument("--config", default="configs/optical_sar_fusion.yaml")
    parser.add_argument("--output", default="data/outputs/fusion_ablation_eval.json")
    args = parser.parse_args()

    run_evaluation(
        dataset_name=args.dataset,
        ablation_mode=args.ablation,
        config_path=args.config,
        output_path=args.output
    )
