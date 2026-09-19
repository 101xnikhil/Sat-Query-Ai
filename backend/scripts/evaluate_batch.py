#!/usr/bin/env python3
"""
Offline Batch Evaluation CLI for SatQuery AI.
Enables offline benchmarking runs on datasets like RSVQA, VRSBench, and CDVQA.
"""
import sys
import json
import argparse
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.schemas.query import QueryRequest
from backend.app.controller.engine import ControllerEngine
from backend.app.config import get_settings

def run_cli_batch(input_json_path: str, output_json_path: str):
    input_file = Path(input_json_path)
    if not input_file.exists():
        print(f"Error: Input file {input_json_path} does not exist.")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        items_data = json.load(f)

    if not isinstance(items_data, list):
        print("Error: Input JSON must be a list of query request objects.")
        sys.exit(1)

    settings = get_settings()
    engine = ControllerEngine(settings=settings)
    results = []

    print(f"🚀 Starting SatQuery AI batch evaluation on {len(items_data)} items...")
    for idx, item in enumerate(items_data, 1):
        req = QueryRequest(**item)
        print(f"[{idx}/{len(items_data)}] Processing query: '{req.query}' (Image IDs: {req.image_ids})")
        try:
            res = engine.process_query(req)
            results.append({
                "query": req.query,
                "task": res.task.value,
                "answer": res.answer,
                "confidence": res.confidence_score,
                "metrics": res.computed_metrics,
                "duration_ms": res.trace.total_duration_ms,
                "validation_passed": res.trace.input_validation.passed
            })
        except Exception as e:
            results.append({
                "query": req.query,
                "error": str(e),
                "status": "failed"
            })

    out_file = Path(output_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Batch evaluation completed. Results written to: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI Offline Batch Evaluator")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON containing array of QueryRequests")
    parser.add_argument("--output", "-o", default="data/outputs/batch_results.json", help="Path to write evaluation results")
    args = parser.parse_args()
    run_cli_batch(args.input, args.output)
