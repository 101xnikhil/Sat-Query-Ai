import os
import json
import yaml
from pathlib import Path
import pytest

from satquery.cli import run_batch

def test_batch_runner_fixed_folder_layout(optical_geotiff, sar_geotiff, tmp_path):
    """
    Requirement 7 & Test:
    python -m satquery batch --manifest x.yaml processes cases offline and writes
    answers, GeoJSON, masks, reports, and traces per case in a fixed folder layout.
    """
    manifest_file = tmp_path / "test_manifest.yaml"
    output_dir = tmp_path / "batch_outputs"

    manifest_content = {
        "cases": [
            {
                "id": "case_vqa",
                "query": "Is there a water body in this image?",
                "images": [optical_geotiff]
            },
            {
                "id": "case_fusion",
                "query": "Fuse optical and SAR imagery to delineate surface water and built-up areas.",
                "images": [optical_geotiff, sar_geotiff]
            }
        ]
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.dump(manifest_content, f)

    summary = run_batch(
        manifest_path=str(manifest_file),
        output_dir=str(output_dir)
    )

    assert summary["total_cases"] == 2
    assert summary["passed_cases"] == 2

    # Check summary.json
    summary_file = output_dir / "summary.json"
    assert summary_file.exists()
    with open(summary_file, "r") as f:
        s_data = json.load(f)
        assert s_data["passed_cases"] == 2

    # Verify fixed folder layout for case_vqa
    vqa_dir = output_dir / "case_vqa"
    assert vqa_dir.exists()
    assert (vqa_dir / "answer.txt").exists()
    assert (vqa_dir / "metrics.json").exists()
    assert (vqa_dir / "trace.json").exists()
    assert (vqa_dir / "report.pdf").exists()

    # Verify fixed folder layout for case_fusion
    fusion_dir = output_dir / "case_fusion"
    assert fusion_dir.exists()
    assert (fusion_dir / "answer.txt").exists()
    assert (fusion_dir / "metrics.json").exists()
    assert (fusion_dir / "trace.json").exists()
    assert (fusion_dir / "report.pdf").exists()
    # Check that overlay or mask file was exported if generated
    assert (fusion_dir / "mask.tif").exists() or (fusion_dir / "overlay.geojson").exists()
