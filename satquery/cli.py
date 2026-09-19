import os
import sys
import json
import shutil
import argparse
import uuid
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

from backend.app.config import get_settings
from backend.app.schemas.common import Modality, TaskType, ImageMetadata
from backend.app.schemas.query import QueryRequest, QueryResponse
from backend.app.controller.engine import ControllerEngine
from backend.controller.router import IncompatibleInputError

def run_single_case(
    query: str,
    image_paths: List[str],
    case_id: Optional[str] = None,
    output_case_dir: Optional[Path] = None,
    engine: Optional[ControllerEngine] = None
) -> Dict[str, Any]:
    """Executes a single SatQuery inquiry using ControllerEngine and writes output artifacts."""
    if engine is None:
        engine = ControllerEngine()

    cid = case_id or f"case_{uuid.uuid4().hex[:8]}"
    req = QueryRequest(
        session_id=cid,
        query=query,
        image_ids=image_paths
    )

    resp = engine.process_query(req)

    # If an output directory for this case is specified, export the fixed folder layout
    if output_case_dir:
        output_case_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. answer.txt
        with open(output_case_dir / "answer.txt", "w", encoding="utf-8") as f:
            f.write(resp.answer)

        # 2. metrics.json
        metrics_data = {
            "case_id": cid,
            "query": resp.query,
            "task": resp.task.value,
            "confidence_score": resp.confidence_score,
            "confidence_breakdown": resp.trace.confidence_breakdown,
            "computed_metrics": resp.computed_metrics
        }
        with open(output_case_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        # 3. trace.json
        with open(output_case_dir / "trace.json", "w", encoding="utf-8") as f:
            f.write(resp.trace.model_dump_json(indent=2))

        # 4. overlay.geojson (if layers present)
        for layer in resp.layers:
            if layer.geojson:
                with open(output_case_dir / "overlay.geojson", "w", encoding="utf-8") as f:
                    f.write(layer.geojson.model_dump_json(indent=2))
                break

        # 5. mask.tif (check tool execution records for output raster masks)
        for tc in resp.trace.tool_calls:
            summary = tc.output_summary
            for k in ["change_mask_path", "classification_map_path", "index_map_path"]:
                if k in summary and summary[k] and Path(summary[k]).exists():
                    try:
                        shutil.copyfile(summary[k], output_case_dir / "mask.tif")
                        break
                    except Exception:
                        pass

        # 6. report.pdf (locate generated PDF)
        reports_dir = Path(engine.settings.app.outputs_dir) / "reports"
        src_pdf = reports_dir / f"SatQuery_Report_{cid}.pdf"
        if src_pdf.exists():
            try:
                shutil.copyfile(src_pdf, output_case_dir / "report.pdf")
            except Exception:
                pass

    return {
        "case_id": cid,
        "task": resp.task.value,
        "answer": resp.answer,
        "confidence_score": resp.confidence_score,
        "computed_metrics": resp.computed_metrics,
        "pdf_report_url": resp.pdf_report_url
    }

def run_batch(
    manifest_path: str,
    output_dir: str = "data/outputs/batch_eval",
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes offline batch evaluation over cases declared in manifest YAML.
    Writes outputs in fixed folder layout per case:
      <output_dir>/<case_id>/answer.txt, metrics.json, trace.json, overlay.geojson, mask.tif, report.pdf
    And global summary.json.
    """
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest file '{manifest_path}' does not exist.")

    with open(m_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError(f"No test cases found in manifest '{manifest_path}'.")

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    settings = get_settings(config_path)
    engine = ControllerEngine(settings=settings)

    print(f"==================================================")
    print(f"🛰️  SatQuery AI - Batch Evaluation Runner")
    print(f"Manifest:   {manifest_path} ({len(cases)} case(s))")
    print(f"Output Dir: {output_dir}")
    print(f"==================================================")

    case_summaries = []
    passed_cases = 0
    failed_cases = 0

    for idx, c in enumerate(cases, 1):
        cid = c.get("id", f"case_{idx:03d}")
        q = c.get("query", "")
        imgs = c.get("images", [])
        print(f"[{idx}/{len(cases)}] Processing {cid}: '{q[:50]}...'")

        case_out_dir = out_base / cid
        try:
            res = run_single_case(
                query=q,
                image_paths=imgs,
                case_id=cid,
                output_case_dir=case_out_dir,
                engine=engine
            )
            passed_cases += 1
            case_summaries.append({
                "id": cid,
                "status": "success",
                "task": res["task"],
                "confidence_score": res["confidence_score"],
                "answer": res["answer"],
                "folder": str(case_out_dir)
            })
            print(f"   -> Success (Task: {res['task']}, Confidence: {res['confidence_score'] * 100:.1f}%)")
        except IncompatibleInputError as e:
            failed_cases += 1
            case_summaries.append({
                "id": cid,
                "status": "rejected_incompatible",
                "error": str(e)
            })
            print(f"   -> Rejected (Incompatible: {e})")
        except Exception as e:
            failed_cases += 1
            case_summaries.append({
                "id": cid,
                "status": "error",
                "error": str(e)
            })
            print(f"   -> Error: {e}")

    summary = {
        "manifest": str(manifest_path),
        "total_cases": len(cases),
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "output_dir": str(out_base),
        "cases": case_summaries
    }

    with open(out_base / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"==================================================")
    print(f"Batch Evaluation Completed: {passed_cases}/{len(cases)} passed.")
    print(f"Summary written to: {out_base / 'summary.json'}")
    print(f"==================================================")

    return summary

def main():
    parser = argparse.ArgumentParser(
        prog="python -m satquery",
        description="SatQuery AI - Agentic Remote Sensing Assistant (ISRO/SAC 26167)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run a single SatQuery AI query pipeline")
    run_parser.add_argument("--config", "-c", type=str, default="configs/default_config.yaml")
    run_parser.add_argument("--query", "-q", type=str, default="What is the dominant terrain in this scene?")
    run_parser.add_argument("--images", "-i", nargs="+", default=[], help="Path(s) to GeoTIFF image files")

    # 'batch' subcommand
    batch_parser = subparsers.add_parser("batch", help="Run batch evaluation over a manifest YAML")
    batch_parser.add_argument("--manifest", "-m", type=str, required=True, help="Path to manifest YAML")
    batch_parser.add_argument("--output-dir", "-o", type=str, default="data/outputs/batch_eval", help="Directory for case outputs")
    batch_parser.add_argument("--config", "-c", type=str, default="configs/default_config.yaml")

    args = parser.parse_args()

    if args.command == "run":
        if not args.images:
            print("Error: Please provide 1 or 2 images via --images", file=sys.stderr)
            sys.exit(1)
        res = run_single_case(query=args.query, image_paths=args.images)
        print(json.dumps(res, indent=2))

    elif args.command == "batch":
        try:
            run_batch(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                config_path=args.config
            )
        except Exception as e:
            print(f"Batch Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
