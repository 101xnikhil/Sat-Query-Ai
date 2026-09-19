import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from ..schemas.common import Modality, TaskType, ImageMetadata
from ..schemas.query import QueryRequest, QueryResponse, LayerItem
from ..schemas.tools import (
    RSVQAOutput, RSCaptionOutput, RSGroundingOutput,
    ChangeVQAOutput, ChangeMapOutput, OpticalSARFusionOutput, SpectralIndexOutput
)
from ..config import get_settings
from ..validators.pipeline import validate_task_inputs, ValidationError
from ..trace.logger import TraceLogger
from ..tools.registry import ToolRegistry
from .router import RuleBasedRouter
from .llm_controller import LLMReasoningController
from ..reporting.pdf_exporter import SatellitePDFReportGenerator
from ..ingestion.reader import read_image_metadata

class ControllerEngine:
    """
    Central orchestrator coordinating:
    Validation -> Routing -> Tool Execution -> Evidence Fusion -> Trace Finalization -> PDF Generation.
    """
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.router = RuleBasedRouter()
        self.registry = ToolRegistry.get_instance()
        self.reasoning_controller = LLMReasoningController()
        reports_dir = Path(self.settings.app.outputs_dir) / "reports"
        self.pdf_generator = SatellitePDFReportGenerator(output_dir=str(reports_dir))
        self._session_cache: Dict[str, QueryResponse] = {}

    def get_cached_response(self, session_id: str) -> Optional[QueryResponse]:
        return self._session_cache.get(session_id)

    def process_query(self, request: QueryRequest) -> QueryResponse:
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:10]}"
        
        # 1. Ingest image metadata for requested images
        images: List[ImageMetadata] = []
        storage_dir = Path(self.settings.app.storage_dir)
        
        for img_id in request.image_ids:
            # Look in storage dir or test fixtures
            found = False
            for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
                candidate = storage_dir / f"{img_id}{ext}"
                if candidate.exists():
                    images.append(read_image_metadata(str(candidate), image_id=img_id))
                    found = True
                    break
            
            # Direct path fallback
            if not found and Path(img_id).exists():
                images.append(read_image_metadata(img_id))
                found = True

            if not found:
                raise FileNotFoundError(f"Image ID '{img_id}' could not be located.")

        # 2. Task classification
        task = self.router.classify_task(
            query=request.query,
            images=images,
            task_override=request.task_override
        )

        # 3. Initialize Observable Trace Logger
        trace_logger = TraceLogger(session_id=session_id, task=task, output_dir=self.settings.app.outputs_dir)

        # 4. Input validation & SAR preprocessing
        passed, val_record, validated_images = validate_task_inputs(
            task=task,
            images=images,
            settings=self.settings
        )
        trace_logger.log_validation(val_record)

        if not passed:
            err_msg = "; ".join(val_record.errors)
            trace = trace_logger.finalize(
                computed_metrics={},
                confidence_score=0.0,
                confidence_breakdown={"validation_error": 0.0}
            )
            return QueryResponse(
                session_id=session_id,
                query=request.query,
                task=task,
                answer=f"Validation Failed: {err_msg}",
                layers=[],
                computed_metrics={"validation_passed": False, "errors": val_record.errors},
                confidence_score=0.0,
                trace=trace
            )

        # 5. Tool Sequence Planning
        plan = self.router.plan_tools(task=task, query=request.query, images=validated_images)

        # 6. Execute Tools
        tool_results: Dict[str, Any] = {}
        layers: List[LayerItem] = []
        computed_metrics: Dict[str, Any] = {}

        for tool_name, params in plan:
            t0 = time.time()
            status = "success"
            err_str = None
            res = None
            try:
                res = self.registry.execute(tool_name, params)
                tool_results[tool_name] = res
            except Exception as e:
                status = "failed"
                err_str = str(e)
            duration_ms = (time.time() - t0) * 1000.0

            summary = res.model_dump() if hasattr(res, "model_dump") else {}
            trace_logger.log_tool_call(
                tool_name=tool_name,
                parameters=params,
                status=status,
                duration_ms=duration_ms,
                output_summary={k: v for k, v in summary.items() if k not in ["geojson", "mask_url"]},
                error=err_str
            )

        # 7. Evidence Fusion & Metrics Synthesis (Rule: Numbers computed from masks)
        final_answer = ""
        base_confidence = 0.85
        conf_breakdown: Dict[str, float] = {}

        # Grounding
        if "rs_grounding" in tool_results:
            ground_res: RSGroundingOutput = tool_results["rs_grounding"]
            final_answer = (
                f"Identified {ground_res.detected_count} target region(s) matching '{request.query}' "
                f"with average detection confidence of {round(sum(ground_res.confidences)/max(1, len(ground_res.confidences)), 3)}."
            )
            computed_metrics["detected_count"] = ground_res.detected_count
            computed_metrics["labels"] = ground_res.labels
            base_confidence = float(np.mean(ground_res.confidences)) if ground_res.confidences else 0.88
            conf_breakdown["grounding"] = base_confidence

            if ground_res.geojson:
                layers.append(LayerItem(
                    layer_id=f"grounding_{uuid.uuid4().hex[:6]}",
                    name="Target Grounding Detections",
                    type="geojson",
                    geojson=ground_res.geojson,
                    color="#00FFCC",
                    opacity=0.8
                ))

        # Optical + SAR Fusion
        elif "optical_sar_fusion" in tool_results:
            fusion_res: OpticalSARFusionOutput = tool_results["optical_sar_fusion"]
            final_answer = (
                f"Cross-modal Optical+SAR two-stream analysis detected:\n"
                f"• Surface Water: {fusion_res.water_area_km2:.4f} km² ({fusion_res.water_percentage:.2f}% of scene)\n"
                f"• Built-up Urban: {fusion_res.built_up_area_km2:.4f} km² ({fusion_res.built_up_percentage:.2f}% of scene)"
            )
            computed_metrics["water_area_km2"] = fusion_res.water_area_km2
            computed_metrics["built_up_area_km2"] = fusion_res.built_up_area_km2
            computed_metrics["water_percentage"] = fusion_res.water_percentage
            computed_metrics["built_up_percentage"] = fusion_res.built_up_percentage

            base_confidence = 0.90
            conf_breakdown["fusion_segmenter"] = 0.90

            # Cross-tool agreement check with spectral_index
            if "spectral_index" in tool_results:
                spec_res: SpectralIndexOutput = tool_results["spectral_index"]
                agreement = 1.0
                try:
                    import rasterio
                    if Path(fusion_res.classification_map_path).exists() and Path(spec_res.index_map_path).exists():
                        with rasterio.open(fusion_res.classification_map_path) as f_src, rasterio.open(spec_res.index_map_path) as s_src:
                            f_arr = f_src.read(1)
                            s_arr = s_src.read(1)
                            mh = min(f_arr.shape[0], s_arr.shape[0])
                            mw = min(f_arr.shape[1], s_arr.shape[1])
                            f_water = (f_arr[:mh, :mw] == 1)
                            s_water = (s_arr[:mh, :mw] > 0.1)
                            inter = np.sum(f_water & s_water)
                            uni = np.sum(f_water | s_water)
                            if uni > 0:
                                agreement = float(inter) / float(uni)
                            else:
                                agreement = 1.0
                    else:
                        pct_diff = abs(fusion_res.water_percentage - spec_res.positive_percentage)
                        agreement = max(0.0, 1.0 - (pct_diff / 100.0))
                except Exception:
                    pct_diff = abs(fusion_res.water_percentage - spec_res.positive_percentage)
                    agreement = max(0.0, 1.0 - (pct_diff / 100.0))

                fused_conf = 0.6 * base_confidence + 0.4 * agreement
                computed_metrics["cross_tool_ndwi_agreement"] = round(agreement, 4)
                conf_breakdown["cross_tool_ndwi_agreement"] = round(agreement, 4)
                final_answer += f"\n• Spectral Cross-Check: NDWI spatial agreement score is {round(agreement*100, 1)}%."
                base_confidence = fused_conf

            if fusion_res.geojson:
                layers.append(LayerItem(
                    layer_id=f"fusion_{uuid.uuid4().hex[:6]}",
                    name="Optical+SAR Classified Zones",
                    type="geojson",
                    geojson=fusion_res.geojson,
                    color="#00E5FF",
                    opacity=0.75
                ))

        # Change Tasks
        elif "change_vqa" in tool_results or "change_map" in tool_results:
            vqa_ans = tool_results["change_vqa"].answer if "change_vqa" in tool_results else ""
            if "change_map" in tool_results:
                cmap_res: ChangeMapOutput = tool_results["change_map"]
                computed_metrics["area_changed_km2"] = cmap_res.area_changed_km2
                computed_metrics["percentage_changed"] = cmap_res.percentage_changed
                computed_metrics["direction_of_change"] = cmap_res.direction_of_change

                metric_stmt = (
                    f"Pixel-level change verification confirms {cmap_res.area_changed_km2:.4f} km² "
                    f"({cmap_res.percentage_changed:.2f}% of the scene) changed. "
                    f"Primary change dynamic: {cmap_res.direction_of_change}."
                )
                final_answer = f"{vqa_ans}\n\n{metric_stmt}" if vqa_ans else metric_stmt
                base_confidence = 0.89
                conf_breakdown["change_detector"] = 0.89
                if cmap_res.geojson:
                    layers.append(LayerItem(
                        layer_id=f"change_{uuid.uuid4().hex[:6]}",
                        name="Temporal Change Footprint",
                        type="geojson",
                        geojson=cmap_res.geojson,
                        color="#FF1744",
                        opacity=0.65
                    ))
            else:
                final_answer = vqa_ans
                base_confidence = 0.86
                conf_breakdown["vqa"] = 0.86

        # Caption
        elif "rs_caption" in tool_results:
            cap_res: RSCaptionOutput = tool_results["rs_caption"]
            final_answer = cap_res.caption
            computed_metrics["tags"] = cap_res.tags
            base_confidence = cap_res.confidence
            conf_breakdown["captioner"] = cap_res.confidence

        # Single VQA
        elif "rs_vqa" in tool_results:
            vqa_res: RSVQAOutput = tool_results["rs_vqa"]
            final_answer = vqa_res.answer
            base_confidence = vqa_res.confidence
            conf_breakdown["vlm"] = vqa_res.confidence

            if "spectral_index" in tool_results:
                spec_res: SpectralIndexOutput = tool_results["spectral_index"]
                final_answer += (
                    f"\n\nSpectral {spec_res.index_type} calculation confirms positive area coverage of "
                    f"{spec_res.positive_area_km2:.4f} km² ({spec_res.positive_percentage:.2f}% of area) "
                    f"with mean index value {spec_res.mean_index:.3f}."
                )
                computed_metrics["spectral_index_mean"] = spec_res.mean_index
                computed_metrics["spectral_area_km2"] = spec_res.positive_area_km2
                computed_metrics["spectral_percentage"] = spec_res.positive_percentage
                if spec_res.geojson:
                    layers.append(LayerItem(
                        layer_id=f"spec_{uuid.uuid4().hex[:6]}",
                        name=f"{spec_res.index_type} Bounding Layer",
                        type="geojson",
                        geojson=spec_res.geojson,
                        color="#76FF03",
                        opacity=0.65
                    ))

        # 8. Optional Analytical Synthesis via LLMReasoningController
        analytical_keywords = ["analyze", "assess", "briefing", "report", "evaluate", "breakdown", "synthesize", "detail", "recommend"]
        if any(k in request.query.lower() for k in analytical_keywords) and computed_metrics:
            try:
                synthesis = self.reasoning_controller.synthesize_analysis(
                    query=request.query,
                    task=task,
                    tool_results=tool_results,
                    computed_metrics=computed_metrics,
                    trace_steps=trace_logger.get_records()
                )
                if synthesis.get("detailed_analysis"):
                    final_answer = f"{final_answer}\n\n**Mission Intelligence Briefing:**\n{synthesis['detailed_analysis']}"
            except Exception:
                pass

        # 9. Finalize Trace
        trace = trace_logger.finalize(
            computed_metrics=computed_metrics,
            confidence_score=base_confidence,
            confidence_breakdown=conf_breakdown
        )

        response = QueryResponse(
            session_id=session_id,
            query=request.query,
            task=task,
            answer=final_answer,
            layers=layers,
            computed_metrics=computed_metrics,
            confidence_score=round(base_confidence, 4),
            trace=trace,
            pdf_report_url=f"/api/reports/{session_id}/pdf"
        )

        # 10. Generate PDF Report & Cache
        try:
            self.pdf_generator.generate_report(response)
        except Exception:
            pass

        self._session_cache[session_id] = response
        return response
