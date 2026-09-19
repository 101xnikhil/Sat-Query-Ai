import os
import sys
import json
import argparse
import uuid
from pathlib import Path
from typing import List, Optional

from backend.app.config import get_settings
from backend.app.schemas.common import Modality, TaskType, ImageMetadata
from backend.app.schemas.query import QueryRequest, QueryResponse
from backend.ingest.reader import read_geotiff_metadata
from backend.controller.router import RuleBasedRouter, IncompatibleInputError
from backend.validators.validators import validate_all
from backend.tools.registry import ToolRegistry
from backend.trace.logger import TraceLogger

def run_pipeline(
    config_path: str,
    query: str,
    image_paths: List[str],
    modality_override: Optional[str] = None,
    benchmark_mode: bool = False
) -> dict:
    """Executes the complete SatQuery AI Phase 1 pipeline from CLI."""
    settings = get_settings(config_path)
    run_id = f"cli_{uuid.uuid4().hex[:8]}"

    print(f"==================================================")
    print(f"🛰️  SatQuery AI CLI - Run ID: {run_id}")
    print(f"Config: {config_path}")
    print(f"Query:  {query}")
    print(f"Images: {image_paths}")
    print(f"==================================================")

    # 1. Ingestion
    images: List[ImageMetadata] = []
    mod_override = None
    if modality_override and modality_override.lower() in [m.value for m in Modality]:
        mod_override = Modality(modality_override.lower())

    for idx, img_p in enumerate(image_paths):
        meta = read_geotiff_metadata(
            file_path=img_p,
            image_id=f"img_{idx+1}_{Path(img_p).stem}",
            modality_override=mod_override,
            benchmark_mode=benchmark_mode
        )
        images.append(meta)
        print(f"-> Ingested: {meta.filename} | Modality: {meta.modality.value} (conf: {meta.modality_confidence}) | Size: {meta.width}x{meta.height} | CRS: {meta.crs}")

    # 2. Routing
    router = RuleBasedRouter()
    task = router.classify_task(query=query, images=images)
    print(f"-> Classified Task: {task.value}")

    # 3. Validation
    val_res, val_rec, validated_images = validate_all(
        task=task,
        images=images,
        settings=settings,
        output_dir=settings.app.outputs_dir
    )

    trace_logger = TraceLogger(
        run_id=run_id,
        task=task,
        query=query,
        output_dir=settings.app.outputs_dir
    )
    trace_logger.log_validation(val_rec)

    if not val_res.ok:
        print(f"❌ Input Validation Failed: {val_res.errors}")
        trace = trace_logger.finalize(computed_metrics={}, confidence_score=None)
        return {
            "run_id": run_id,
            "status": "validation_failed",
            "errors": val_res.errors,
            "trace_file": str(Path(settings.app.outputs_dir) / f"{run_id}_trace.json")
        }

    print(f"-> Validation Passed. Actions: {val_res.actions_taken}")

    # 4. Tool Planning & Execution
    plan = router.plan_tools(task=task, query=query, images=validated_images)
    registry = ToolRegistry.get_instance()
    tool_outputs = {}

    import time
    for tool_name, params in plan:
        t0 = time.time()
        print(f"   * Executing Tool: {tool_name} with params {params}")
        output = registry.execute(tool_name, params)
        dur_ms = (time.time() - t0) * 1000.0
        out_dict = output.model_dump()
        tool_outputs[tool_name] = out_dict
        trace_logger.log_tool_call(
            tool_name=tool_name,
            parameters=params,
            status="success",
            duration_ms=dur_ms,
            output_summary=out_dict
        )

    # 5. Trace Finalization
    trace = trace_logger.finalize(
        computed_metrics={"tools_executed": list(tool_outputs.keys())},
        confidence_score=None
    )
    trace_path = Path(settings.app.outputs_dir) / f"{run_id}_trace.json"

    # Extract primary answer
    primary_answer = ""
    for t_out in tool_outputs.values():
        if "answer" in t_out:
            primary_answer = t_out["answer"]
            break
        elif "caption" in t_out:
            primary_answer = t_out["caption"]
            break

    print(f"==================================================")
    print(f"✅ Execution Completed Successfully!")
    print(f"Answer:     {primary_answer}")
    print(f"Trace File: {trace_path}")
    print(f"==================================================")

    return {
        "run_id": run_id,
        "task": task.value,
        "answer": primary_answer,
        "tool_outputs": tool_outputs,
        "trace_file": str(trace_path)
    }

def main():
    parser = argparse.ArgumentParser(
        prog="python -m satquery",
        description="SatQuery AI - Agentic Remote Sensing Assistant (ISRO/SAC 26167)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run the SatQuery AI query pipeline")
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "--query", "-q",
        type=str,
        default="What is the dominant terrain or land cover in this scene?",
        help="Natural language remote sensing query"
    )
    run_parser.add_argument(
        "--images", "-i",
        nargs="+",
        default=[],
        help="Path(s) to 1 or 2 remote sensing GeoTIFF imagery files"
    )
    run_parser.add_argument(
        "--modality", "-m",
        type=str,
        choices=["optical", "multispectral", "sar"],
        default=None,
        help="Explicit modality override"
    )
    run_parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Allow non-georeferenced PNG/JPEG benchmark imagery"
    )

    args = parser.parse_args()

    if args.command == "run":
        if not args.images:
            print("No images provided. Please provide 1 or 2 GeoTIFF files via --images path/to/image.tif")
            sys.exit(1)
        try:
            result = run_pipeline(
                config_path=args.config,
                query=args.query,
                image_paths=args.images,
                modality_override=args.modality,
                benchmark_mode=args.benchmark_mode
            )
            print(json.dumps(result, indent=2))
        except IncompatibleInputError as e:
            print(f"Error (Incompatible Input): {e}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            print(f"Execution Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
