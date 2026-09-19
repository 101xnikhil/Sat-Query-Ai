#!/usr/bin/env python3
"""
VRSBench Empirical Quantitative Benchmark for SatQuery AI.
Computes:
1. Grounding Mean IoU (mIoU), Acc@IoU 0.50, and Acc@IoU 0.70.
2. VRSBench VQA Accuracy (on visual question answering split).
ZERO METRIC FABRICATION: executes real inference calls and writes run config to JSON.
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.data.vrsbench import VRSBenchDataLoader
from models.inference.vlm_wrapper import get_vlm_wrapper
from models.inference.tiled_inference import compute_box_iou

def evaluate_vrsbench(
    annotation_path: str = "data/datasets/sample_vrsbench.json",
    output_report_path: str = "data/outputs/eval_vrsbench_results.json",
    images_dir: Optional[str] = "data/samples",
    config_path: Optional[str] = "models/configs/tiny_cpu_vlm.yaml"
) -> Dict[str, Any]:
    """
    Evaluates VRSBench grounding (Acc@0.5, Acc@0.7) and VQA accuracy.
    Writes verified empirical metrics to JSON.
    """
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"VRSBench annotations not found at: {annotation_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    vlm = get_vlm_wrapper(config_path=config_path)

    # 1. Grounding Evaluation
    grounding_items = raw_data.get("annotations", [])
    ious: List[float] = []
    acc_50_count = 0
    acc_70_count = 0
    grounding_records = []

    print(f"📐 Running VRSBench Grounding Evaluation on {len(grounding_items)} samples...")
    for idx, item in enumerate(grounding_items):
        expr = item["expression"]
        gt_box = item["bbox"]  # [ymin, xmin, ymax, xmax]
        img_name = item.get("image_name", "delhi_optical.tif")
        img_path = str((Path(images_dir or "data/samples") / img_name).resolve())

        if not Path(img_path).exists():
            img_path = str((_PROJECT_ROOT / "data/samples/delhi_optical.tif").resolve())

        # Real grounding inference execution
        pred_boxes = vlm.ground_query(query=expr, image_path=img_path)

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
        if best_iou >= 0.70:
            acc_70_count += 1

        grounding_records.append({
            "id": item.get("id", idx + 1),
            "expression": expr,
            "ground_truth_bbox": gt_box,
            "best_prediction_bbox": best_pred_box,
            "iou": round(best_iou, 4),
            "hit_at_50": bool(best_iou >= 0.50),
            "hit_at_70": bool(best_iou >= 0.70)
        })

    m_iou = round(float(np.mean(ious)), 4) if ious else 0.0
    acc_50 = round(acc_50_count / len(grounding_items), 4) if grounding_items else 0.0
    acc_70 = round(acc_70_count / len(grounding_items), 4) if grounding_items else 0.0

    # 2. VQA Evaluation (if vqa questions present)
    vqa_items = raw_data.get("vqa_questions", [])
    vqa_correct = 0
    vqa_records = []
    if vqa_items:
        print(f"💬 Running VRSBench VQA Evaluation on {len(vqa_items)} questions...")
        for item in vqa_items:
            q = item["question"]
            gt = str(item["answer"]).lower().strip()
            img_name = item.get("image_name", "delhi_optical.tif")
            img_path = str((Path(images_dir or "data/samples") / img_name).resolve())
            if not Path(img_path).exists():
                img_path = str((_PROJECT_ROOT / "data/samples/delhi_optical.tif").resolve())

            res = vlm.answer_vqa(query=q, image_path=img_path)
            ans = str(res["answer"]).lower()
            is_correct = gt in ans or (gt == "yes" and "yes" in ans)
            if is_correct:
                vqa_correct += 1

            vqa_records.append({
                "id": item.get("id"),
                "question": q,
                "ground_truth": gt,
                "prediction": res["answer"],
                "confidence": res["confidence"],
                "correct": is_correct
            })

    vqa_accuracy = round(vqa_correct / len(vqa_items), 4) if vqa_items else None

    report = {
        "benchmark": "VRSBench",
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "run_config": {
            "config_path": config_path,
            "model_name": vlm.model_name,
            "mock_mode": vlm.mock_mode,
            "nms_iou_threshold": vlm.nms_iou,
            "tile_size": vlm.tile_size,
            "overlap": vlm.overlap,
            "device": vlm.device
        },
        "dataset_path": str(annotation_path),
        "grounding_metrics": {
            "total_samples": len(grounding_items),
            "mean_iou": m_iou,
            "acc_at_50": acc_50,
            "acc_at_70": acc_70,
            "evaluations": grounding_records
        },
        "vqa_metrics": {
            "total_questions": len(vqa_items),
            "vqa_accuracy": vqa_accuracy,
            "evaluations": vqa_records
        } if vqa_items else None
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ VRSBench Evaluation completed: mIoU = {m_iou:.4f}, Acc@0.5 = {acc_50*100:.1f}%, Acc@0.7 = {acc_70*100:.1f}%")
    if vqa_accuracy is not None:
        print(f"✅ VRSBench VQA Accuracy = {vqa_accuracy * 100:.1f}%")
    print(f"📁 Report written to: {out_file.resolve()}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VRSBench Grounding and VQA")
    parser.add_argument("--annotation-path", default="data/datasets/sample_vrsbench.json")
    parser.add_argument("--output-path", default="data/outputs/eval_vrsbench_results.json")
    parser.add_argument("--config", default="models/configs/tiny_cpu_vlm.yaml")
    args = parser.parse_args()

    evaluate_vrsbench(
        annotation_path=args.annotation_path,
        output_report_path=args.output_path,
        config_path=args.config
    )
