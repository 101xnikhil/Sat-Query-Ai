#!/usr/bin/env python3
"""
CDVQA (Change Detection Visual Question Answering) Quantitative Benchmark.
Evaluates bi-temporal question answering accuracy, change detection consistency,
and empirical mask area metrics.
ZERO METRIC FABRICATION: executes real bi-temporal inference on provided evaluation splits.
"""
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add root directory to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from models.training.dataset_loaders import CDVQADataLoader
from backend.app.tools.change_vqa import ChangeVQATool
from backend.app.tools.change_map import ChangeMapTool
from backend.app.schemas.tools import ChangeVQAInput, ChangeMapInput

def normalize_text(text: str) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    if s in ["y", "true"]:
        return "yes"
    if s in ["n", "false"]:
        return "no"
    return s

def check_cdvqa_match(pred: str, gt: str) -> bool:
    norm_pred = normalize_text(pred)
    norm_gt = normalize_text(gt)
    if norm_pred == norm_gt:
        return True
    if re.search(rf"\b{re.escape(norm_gt)}\b", norm_pred):
        return True
    words = [w for w in norm_gt.split() if len(w) > 2 and w not in ["and", "the", "for"]]
    if words and all(re.search(rf"\b{re.escape(w)}\b", norm_pred) for w in words):
        return True
    return False

def evaluate_cdvqa(
    annotation_path: str = "data/datasets/sample_cdvqa.json",
    output_report_path: str = "data/outputs/eval_cdvqa_results.json",
    images_dir: Optional[str] = "data/samples"
) -> Dict[str, Any]:
    loader = CDVQADataLoader(annotation_path=annotation_path, images_dir=images_dir)
    if len(loader) == 0:
        raise FileNotFoundError(f"No samples found at annotation path: {annotation_path}")

    change_vqa_tool = ChangeVQATool()
    change_map_tool = ChangeMapTool()

    results = []
    correct_vqa = 0
    correct_detection = 0
    confidences: List[float] = []
    category_stats: Dict[str, Dict[str, int]] = {}

    print(f"🛰️  Running CDVQA Evaluation on {len(loader)} bi-temporal samples...")
    for idx, sample in enumerate(loader):
        q = sample["question"]
        gt_ans = sample["answer"]
        c_type = sample["change_type"]
        expected_has_change = sample["has_change"]

        t1_path = sample["image_t1_path"]
        t2_path = sample["image_t2_path"]

        # 1. Execute Real Change-VQA
        vqa_out = change_vqa_tool.run(ChangeVQAInput(
            query=q,
            image_t1_path=t1_path,
            image_t2_path=t2_path
        ))
        pred_ans = vqa_out.answer
        conf = vqa_out.confidence
        confidences.append(conf)

        is_vqa_correct = check_cdvqa_match(pred_ans, gt_ans)
        if is_vqa_correct:
            correct_vqa += 1

        # 2. Execute Real Change Map
        cmap_out = change_map_tool.run(ChangeMapInput(
            image_t1_path=t1_path,
            image_t2_path=t2_path,
            threshold=0.45
        ))
        detected_has_change = cmap_out.area_changed_km2 > 0.0
        is_detection_correct = (detected_has_change == expected_has_change)
        if is_detection_correct:
            correct_detection += 1

        # Track per-category accuracy
        if c_type not in category_stats:
            category_stats[c_type] = {"correct": 0, "total": 0}
        category_stats[c_type]["total"] += 1
        if is_vqa_correct:
            category_stats[c_type]["correct"] += 1

        results.append({
            "id": sample["id"],
            "question": q,
            "ground_truth": gt_ans,
            "prediction": pred_ans,
            "confidence": conf,
            "vqa_correct": is_vqa_correct,
            "category": c_type,
            "expected_change": expected_has_change,
            "detected_change": detected_has_change,
            "area_changed_km2": cmap_out.area_changed_km2,
            "percentage_changed": cmap_out.percentage_changed,
            "direction": cmap_out.direction_of_change
        })

    vqa_accuracy = round(correct_vqa / len(loader), 4)
    detection_accuracy = round(correct_detection / len(loader), 4)
    mean_conf = round(float(np.mean(confidences)), 4)

    cat_acc = {
        cat: round(data["correct"] / data["total"], 4)
        for cat, data in category_stats.items()
    }

    report = {
        "benchmark": "CDVQA",
        "dataset_path": annotation_path,
        "total_samples": len(loader),
        "vqa_accuracy": vqa_accuracy,
        "change_detection_accuracy": detection_accuracy,
        "mean_confidence": mean_conf,
        "category_accuracy": cat_acc,
        "sample_evaluations": results
    }

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("==================================================")
    print(f"✅ CDVQA VQA Accuracy:             {vqa_accuracy * 100:.2f}% ({correct_vqa}/{len(loader)})")
    print(f"🎯 Change Detection Consistency:  {detection_accuracy * 100:.2f}% ({correct_detection}/{len(loader)})")
    print(f"📊 Mean Confidence:                {mean_conf * 100:.2f}%")
    print(f"📋 Category Accuracy:              {cat_acc}")
    print(f"📁 Report saved to:                 {output_report_path}")
    print("==================================================")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on CDVQA Benchmark")
    parser.add_argument("--annotations", "-a", default="data/datasets/sample_cdvqa.json")
    parser.add_argument("--images-dir", "-i", default="data/samples")
    parser.add_argument("--output", "-o", default="data/outputs/eval_cdvqa_results.json")
    args = parser.parse_args()

    evaluate_cdvqa(
        annotation_path=args.annotations,
        output_report_path=args.output,
        images_dir=args.images_dir
    )
