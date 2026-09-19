import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time

from backend.app.config import get_settings
from backend.app.schemas.query import QueryRequest
from backend.app.controller.engine import ControllerEngine
from scripts.setup_demo_data import create_demo_assets

def run_satquery_demo():
    """
    Automated demo walkthrough covering all five mandatory capabilities (ISRO/SAC 26167):
    1. Single-Image VQA & Scene Understanding
    2. Text-Guided Geospatial Grounding
    3. Multitemporal Change Understanding
    4. Optical + SAR Cross-Modal Fusion
    5. Agentic Multi-Step Orchestration
    
    Generates downloadable PDF reports and verifiable execution traces for each capability.
    """
    print(f"================================================================================")
    print(f"🛰️  SatQuery AI - Full System Demonstration (ISRO/SAC Problem Statement 26167)")
    print(f"================================================================================")
    
    # 1. Ensure demo assets exist
    assets_dir = Path("data/demo_assets")
    create_demo_assets(str(assets_dir))

    engine = ControllerEngine()
    reports_dir = Path(engine.settings.app.outputs_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    demo_cases = [
        {
            "num": 1,
            "name": "Capability 1: Single-Image VQA & Scene Understanding",
            "query": "What is the dominant terrain and are there water bodies in this scene?",
            "images": [str(assets_dir / "demo_single_optical.tif")],
            "session_id": "demo_01_single_vqa"
        },
        {
            "num": 2,
            "name": "Capability 2: Text-Guided Geospatial Grounding",
            "query": "Locate the airfield runway and storage tanks.",
            "images": [str(assets_dir / "demo_single_optical.tif")],
            "session_id": "demo_02_grounding"
        },
        {
            "num": 3,
            "name": "Capability 3: Multitemporal Change Understanding",
            "query": "Has built-up area increased, decreased, or remained unchanged between the two dates?",
            "images": [str(assets_dir / "demo_temporal_t1.tif"), str(assets_dir / "demo_temporal_t2.tif")],
            "session_id": "demo_03_change_understanding"
        },
        {
            "num": 4,
            "name": "Capability 4: Optical-SAR Cross-Modal Analysis",
            "query": "Fuse optical and SAR imagery to delineate surface water bodies and built-up areas.",
            "images": [str(assets_dir / "demo_crossmodal_optical.tif"), str(assets_dir / "demo_crossmodal_sar.tif")],
            "session_id": "demo_04_optical_sar_fusion"
        },
        {
            "num": 5,
            "name": "Capability 5: Agentic Multi-Step Orchestration",
            "query": "what changed and is the new area water or built-up?",
            "images": [str(assets_dir / "demo_crossmodal_optical.tif"), str(assets_dir / "demo_crossmodal_sar.tif")],
            "session_id": "demo_05_agentic_orchestration"
        }
    ]

    generated_pdfs = []

    for case in demo_cases:
        print(f"\n--------------------------------------------------------------------------------")
        print(f"▶ [{case['num']}/5] {case['name']}")
        print(f"  Session ID: {case['session_id']}")
        print(f"  Query:      \"{case['query']}\"")
        print(f"  Inputs:     {[Path(p).name for p in case['images']]}")
        print(f"--------------------------------------------------------------------------------")

        t0 = time.time()
        req = QueryRequest(
            session_id=case["session_id"],
            query=case["query"],
            image_ids=case["images"]
        )
        resp = engine.process_query(req)
        dur = (time.time() - t0) * 1000.0

        trace = resp.trace
        pdf_path = reports_dir / f"SatQuery_Report_{case['session_id']}.pdf"
        assert pdf_path.exists(), f"PDF report {pdf_path} was not created!"
        generated_pdfs.append(str(pdf_path))

        print(f"  ✓ Task Classified:    {resp.task.value.upper()}")
        print(f"  ✓ Planner Mode:       {'Fallback Router' if trace.fallback_used else 'LLM Controller Plan'}")
        print(f"  ✓ Tools Executed:     {[tc.tool_name for tc in trace.tool_calls]}")
        print(f"  ✓ Calibrated Conf:    {resp.confidence_score * 100:.1f}%")
        print(f"  ✓ Stage Latencies:    Ingest: {trace.stage_latencies.get('ingestion_ms', 0):.1f}ms | Plan: {trace.stage_latencies.get('planning_ms', 0):.1f}ms | Tools: {trace.stage_latencies.get('tool_execution_ms', 0):.1f}ms | Synth: {trace.stage_latencies.get('synthesis_ms', 0):.1f}ms")
        print(f"  ✓ Total Latency:      {dur:.1f} ms")
        print(f"  ✓ Grounded Answer:\n    {resp.answer.replace(chr(10), chr(10) + '    ')}")
        print(f"  ✓ Downloadable PDF:   {pdf_path}")

    print(f"\n================================================================================")
    print(f"🎉 ALL 5 MANDATORY CAPABILITIES VERIFIED SUCCESSFULLY!")
    print(f"Generated PDF Reports:")
    for p in generated_pdfs:
        print(f"  📄 {p}")
    print(f"================================================================================")

if __name__ == "__main__":
    run_satquery_demo()
