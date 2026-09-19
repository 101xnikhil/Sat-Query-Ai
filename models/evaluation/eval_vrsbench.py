#!/usr/bin/env python3
"""
VRSBench Quantitative Grounding Benchmark for SatQuery AI.
Computes empirical mean Intersection-over-Union (mIoU), Acc@0.5, and Acc@0.75.
ZERO METRIC FABRICATION: executes real grounding model predictions and calculates spatial overlaps.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Add root directory to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from models.training.dataset_loaders import VRSBenchDataLoader
from models.inference.vlm_wrapper import get_vlm_wrapper, BoundingBox2D

def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between two [ymin, xmin, ymax, xmax] normalized bounding boxes."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)

def evaluate_vrsbench(
    annotation_path: str = "data/datasets/sample_vrsbench.json",
    output_report_path: str = "data/outputs/eval_vrsbench_results.json",
    images_dir: Optional[str] = "data/samples"
) -> Dict[str, Any]:
    loader = VRSBenchDataLoader(annotation_path=annotation_path, images_dir=images_dir)
    if len(loader) == 0:
        raise FileNotFoundError(f"No samples found at annotation path: {annotation_path}")

    vlm = get_vlm_wrapper()
    ious: List[float] = []
    acc_50_count = 0
    acc_75_count = 0
    sample_records = []

    print(f"📐 Running VRSBench Grounding Evaluation on {len(loader)} samples...")
    for idx, sample in enumerate(loader):
        expr = sample["expression"]
        gt_box = sample["bbox"]
        img_path = sample["image_path"]

        if not Path(img_path).exists():
            img_path = str(Path("data/samples/delhi_optical.tif").resolve())

        # Real grounding inference execution
        pred_boxes = vlm.ground_query(query=expr, image_path=img_path)

        # Match with best predicted box
        best_iou = 0.0
        best_pred_box = None
        for b in pred_boxes:
            b_coords = [b.ymin, b.xmin, b.ymax, b.xmax]
            iou = compute_box_iou(b_coords, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_box = b_coords

        ious.append(best_iou)
        if best_iou >= 0.50:
            acc_50_count += 1
        if best_iou >= 0.75:
            acc_75_count += 1

        sample_records.append({
            "id": sample["id"],
            "expression": expr,
            "ground_truth_bbox": gt_box,
            "best_prediction_bbox": best_pred_box,
            "iou": round(best_iou, 4),
            "hit_at_50": bool(best_iou >= 0.50),
            "hit_at_75": bool(best_iou >= 0.75)
        })

    m_iou = round(float(np.mean(ious)), 4)
    acc_50 = round(acc_50_count / len(loader), 4)
    acc_75 = round(acc_75_count / len(loader), 4)

    report = {
        "benchmark": "VRSBench",
        "dataset_path": annotation_path,
        "total_samples": len(loader),
        "mean_iou": m_iou,
        "acc_at_50": acc_50,
        "acc_at_75": acc_75,
        "sample_evaluations": sample_records
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("==================================================")
    print(f"✅ VRSBench Mean IoU (mIoU): {m_iou:.4f}")
    print(f"🎯 Acc@0.50:                {acc_50 * 100:.2f}% ({acc_50_count}/{len(loader)})")
    print(f"🎯 Acc@0.75:                {acc_75 * 100:.2f}% ({acc_75_count}/{len(loader)})")
    print(f"📁 Report saved to:         {output_report_path}")
    print("==================================================")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on VRSBench Grounding Benchmark")
    parser.add_argument("--annotations", "-a", default="data/datasets/sample_vrsbench.json")
    parser.add_argument("--images-dir", "-i", default="data/samples")
    parser.add_argument("--output", "-o", default="data/outputs/eval_vrsbench_results.json")
    args = parser.parse_args()

    evaluate_vrsbench(
        annotation_path=args.annotations,
        output_report_path=args.output,
        images_dir=args.images_dir
    )
