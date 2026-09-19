import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from ..schemas.common import Modality, TaskType, ImageMetadata, GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
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
from .llm_planner import LLMPlanner
from .llm_controller import LLMReasoningController
from ...confidence.scorer import get_confidence_scorer
from ..cache import get_raster_cache
from ..reporting.pdf_exporter import SatellitePDFReportGenerator
from ..ingestion.reader import read_image_metadata

class ControllerEngine:
    """
    Central orchestrator coordinating:
    Validation -> LLM Planning / Router Fallback -> Tool Execution -> Multi-Step Evidence Fusion -> Trace Finalization -> PDF Generation.
    """
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.router = RuleBasedRouter()
        self.planner = LLMPlanner(settings=self.settings)
        self.registry = ToolRegistry.get_instance()
        self.reasoning_controller = LLMReasoningController()
        self.confidence_scorer = get_confidence_scorer()
        self.raster_cache = get_raster_cache()
        reports_dir = Path(self.settings.app.outputs_dir) / "reports"
        self.pdf_generator = SatellitePDFReportGenerator(output_dir=str(reports_dir))
        self._session_cache: Dict[str, QueryResponse] = {}

    def get_cached_response(self, session_id: str) -> Optional[QueryResponse]:
        return self._session_cache.get(session_id)

    def process_query(self, request: QueryRequest) -> QueryResponse:
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:10]}"
        t_ingest_0 = time.time()
        
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

        ingest_ms = (time.time() - t_ingest_0) * 1000.0

        # 2. Task classification
        task = self.router.classify_task(
            query=request.query,
            images=images,
            task_override=request.task_override
        )

        # 3. Initialize Observable Trace Logger
        trace_logger = TraceLogger(session_id=session_id, task=task, output_dir=self.settings.app.outputs_dir)
        trace_logger.log_stage_latency("ingestion_ms", ingest_ms)

        # 4. Input validation & SAR preprocessing
        t_val_0 = time.time()
        passed, val_record, validated_images = validate_task_inputs(
            task=task,
            images=images,
            settings=self.settings
        )
        val_ms = (time.time() - t_val_0) * 1000.0
        trace_logger.log_validation(val_record)
        trace_logger.log_stage_latency("validation_ms", val_ms)

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

        # 5. Tool Sequence Planning via LLM Planner with Fallback
        t_plan_0 = time.time()
        task, plan, fallback_used, planner_actions = self.planner.plan(
            query=request.query,
            images=validated_images,
            task_override=request.task_override
        )
        plan_ms = (time.time() - t_plan_0) * 1000.0
        trace_logger.log_stage_latency("planning_ms", plan_ms)
        trace_logger.set_fallback_used(fallback_used)
        trace_logger.trace.actions_taken.extend(planner_actions)
        trace_logger.trace.task = task

        # 6. Execute Tools
        t_tools_0 = time.time()
        tool_results: Dict[str, Any] = {}
        layers: List[LayerItem] = []
        computed_metrics: Dict[str, Any] = {}

        for tool_name, params in plan:
            # Pass outputs between chained tools (Requirement 7)
            if tool_name == "change_vqa" and "change_map" in tool_results:
                cmap_res = tool_results["change_map"]
                params["change_mask_path"] = cmap_res.change_mask_path
                params["change_stats"] = {
                    "area_changed_m2": getattr(cmap_res, "area_changed_m2", 0.0),
                    "area_changed_ha": getattr(cmap_res, "area_changed_ha", 0.0),
                    "area_changed_km2": cmap_res.area_changed_km2,
                    "percentage_changed": cmap_res.percentage_changed,
                    "direction_of_change": cmap_res.direction_of_change,
                    "per_class_change": getattr(cmap_res, "per_class_change", {})
                }

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
        tools_ms = (time.time() - t_tools_0) * 1000.0
        trace_logger.log_stage_latency("tool_execution_ms", tools_ms)

        # Multi-step synthesis: compound change and classification (e.g. change_map + optical_sar_fusion)
        if "change_map" in tool_results and "optical_sar_fusion" in tool_results:
            cmap_res = tool_results["change_map"]
            fusion_res = tool_results["optical_sar_fusion"]
            try:
                import rasterio
                if Path(cmap_res.change_mask_path).exists() and Path(fusion_res.classification_map_path).exists():
                    with rasterio.open(cmap_res.change_mask_path) as c_src, rasterio.open(fusion_res.classification_map_path) as f_src:
                        c_mask = c_src.read(1) > 0
                        f_map = f_src.read(1)
                        mh = min(c_mask.shape[0], f_map.shape[0])
                        mw = min(c_mask.shape[1], f_map.shape[1])
                        c_crop = c_mask[:mh, :mw]
                        f_crop = f_map[:mh, :mw]

                        changed_px = int(np.sum(c_crop))
                        ch_water_px = int(np.sum(c_crop & (f_crop == 1)))
                        ch_built_px = int(np.sum(c_crop & (f_crop == 2)))

                        res_x = abs(c_src.transform[0]) or 10.0
                        res_y = abs(c_src.transform[4]) or 10.0
                        px_area = res_x * res_y

                        ch_water_m2 = round(float(ch_water_px * px_area), 2)
                        ch_built_m2 = round(float(ch_built_px * px_area), 2)
                        ch_water_pct = round(float(ch_water_px / max(1, changed_px) * 100.0), 2)
                        ch_built_pct = round(float(ch_built_px / max(1, changed_px) * 100.0), 2)

                        computed_metrics["changed_water_area_m2"] = ch_water_m2
                        computed_metrics["changed_water_percentage"] = ch_water_pct
                        computed_metrics["changed_built_up_area_m2"] = ch_built_m2
                        computed_metrics["changed_built_up_percentage"] = ch_built_pct

                        trace_logger.trace.actions_taken.append(
                            f"Multi-step analysis: Decomposed {changed_px} changed pixels into {ch_water_m2:,.1f} m² water ({ch_water_pct}%) "
                            f"and {ch_built_m2:,.1f} m² built-up ({ch_built_pct}%)."
                        )
            except Exception:
                pass

        # 7. Evidence Fusion & Metrics Synthesis (Rule: Numbers computed from masks)
        t_synth_0 = time.time()
        final_answer = ""
        base_confidence = 0.85
        conf_breakdown: Dict[str, float] = {}

        # Change Tasks
        if task in (TaskType.CHANGE_VQA, TaskType.CHANGE_MAP) or (("change_vqa" in tool_results or "change_map" in tool_results) and "optical_sar_fusion" not in tool_results):
            cvqa_res = tool_results.get("change_vqa")
            vqa_ans = cvqa_res.answer if cvqa_res else ""
            if "change_map" in tool_results:
                cmap_res: ChangeMapOutput = tool_results["change_map"]
                area_m2 = getattr(cmap_res, "area_changed_m2", 0.0)
                area_ha = getattr(cmap_res, "area_changed_ha", round(area_m2 / 10000.0, 4))
                area_km2 = cmap_res.area_changed_km2
                pct = cmap_res.percentage_changed
                direction = cmap_res.direction_of_change

                computed_metrics["area_changed_m2"] = area_m2
                computed_metrics["area_changed_ha"] = area_ha
                computed_metrics["area_changed_km2"] = area_km2
                computed_metrics["percentage_changed"] = pct
                computed_metrics["direction_of_change"] = direction
                if getattr(cmap_res, "per_class_change", None):
                    computed_metrics["per_class_change"] = cmap_res.per_class_change

                if vqa_ans:
                    final_answer = vqa_ans
                else:
                    final_answer = (
                        f"Multitemporal change analysis confirms {area_m2:.1f} m² ({area_ha:.3f} ha, {pct:.2f}% of scene) "
                        f"underwent change. Primary dynamic: {direction}."
                    )

                if "changed_water_area_m2" in computed_metrics and "changed_built_up_area_m2" in computed_metrics:
                    final_answer += (
                        f"\n• Multi-step Land-Cover Breakdown: Of the changed area, {computed_metrics['changed_water_area_m2']:,.1f} m² "
                        f"({computed_metrics['changed_water_percentage']}%) is surface water and "
                        f"{computed_metrics['changed_built_up_area_m2']:,.1f} m² ({computed_metrics['changed_built_up_percentage']}%) "
                        f"is built-up urban structures."
                    )

                if cvqa_res:
                    base_confidence = cvqa_res.confidence
                    conf_breakdown["change_vqa"] = cvqa_res.confidence
                    conf_breakdown["change_detector"] = 0.90
                    if cvqa_res.vlm_cross_check_agreed is not None:
                        computed_metrics["vlm_cross_check_agreed"] = cvqa_res.vlm_cross_check_agreed
                else:
                    base_confidence = 0.89
                    conf_breakdown["change_detector"] = 0.89

                if cmap_res.geojson:
                    layers.append(LayerItem(
                        layer_id=f"change_{uuid.uuid4().hex[:6]}",
                        name="Temporal Change Footprint",
                        type="geojson",
                        geojson=cmap_res.geojson,
                        color="#f43f5e",
                        opacity=0.65
                    ))
            else:
                final_answer = vqa_ans
                base_confidence = cvqa_res.confidence if cvqa_res else 0.86
                conf_breakdown["vqa"] = base_confidence

        # Grounding
        elif "rs_grounding" in tool_results:
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
            
            water_m2 = getattr(fusion_res, "water_area_m2", 0.0)
            if water_m2 == 0.0 and fusion_res.water_area_km2 > 0:
                water_m2 = round(fusion_res.water_area_km2 * 1_000_000.0, 2)
            water_ha = getattr(fusion_res, "water_area_ha", round(water_m2 / 10_000.0, 4))

            built_m2 = getattr(fusion_res, "built_up_area_m2", 0.0)
            if built_m2 == 0.0 and fusion_res.built_up_area_km2 > 0:
                built_m2 = round(fusion_res.built_up_area_km2 * 1_000_000.0, 2)
            built_ha = getattr(fusion_res, "built_up_area_ha", round(built_m2 / 10_000.0, 4))

            computed_metrics["water_area_m2"] = water_m2
            computed_metrics["water_area_ha"] = water_ha
            computed_metrics["water_area_km2"] = fusion_res.water_area_km2
            computed_metrics["water_percentage"] = fusion_res.water_percentage
            computed_metrics["built_up_area_m2"] = built_m2
            computed_metrics["built_up_area_ha"] = built_ha
            computed_metrics["built_up_area_km2"] = fusion_res.built_up_area_km2
            computed_metrics["built_up_percentage"] = fusion_res.built_up_percentage
            computed_metrics["ablation_mode"] = getattr(fusion_res, "ablation_mode", "fused")

            # Handle degradation notes and resampling
            deg_mode = getattr(fusion_res, "degradation_mode", "nominal")
            if deg_mode and deg_mode != "nominal":
                computed_metrics["degradation_mode"] = deg_mode
                trace_logger.trace.actions_taken.append(f"Degradation mode: {deg_mode}")

            resamp_info = getattr(fusion_res, "resampling_info", None)
            if resamp_info:
                computed_metrics["resampling_info"] = resamp_info
                trace_logger.trace.actions_taken.append(
                    f"Auto-resampled SAR ({resamp_info.get('source_shape')}) to Optical grid ({resamp_info.get('target_shape')}) via {resamp_info.get('method', 'bilinear')}"
                )

            final_answer = (
                f"Cross-modal Optical+SAR two-stream analysis detected:\n"
                f"• Surface Water: {water_m2:,.1f} m² ({water_ha:.3f} ha, {fusion_res.water_percentage:.2f}% of scene)\n"
                f"• Built-up Urban: {built_m2:,.1f} m² ({built_ha:.3f} ha, {fusion_res.built_up_percentage:.2f}% of scene)"
            )
            if deg_mode and deg_mode != "nominal":
                final_answer += f"\n• Sensor Mode: {deg_mode}."

            base_confidence = 0.90
            conf_breakdown["fusion_segmenter"] = 0.90

            # Cross-tool agreement check with spectral_index
            if "spectral_index" in tool_results:
                spec_res: SpectralIndexOutput = tool_results["spectral_index"]
                if getattr(spec_res, "status", "computed") == "not_applicable":
                    reason = getattr(spec_res, "reason", "Required bands missing")
                    computed_metrics["cross_tool_ndwi_status"] = "not_applicable"
                    computed_metrics["cross_tool_ndwi_reason"] = reason
                    trace_logger.trace.actions_taken.append(f"Spectral NDWI cross-check bypassed: {reason}")
                    final_answer += f"\n• Spectral Cross-Check: NDWI not applicable ({reason})."
                else:
                    agreement = 1.0
                    try:
                        import rasterio
                        if Path(fusion_res.classification_map_path).exists() and spec_res.index_map_path and Path(spec_res.index_map_path).exists():
                            with rasterio.open(fusion_res.classification_map_path) as f_src, rasterio.open(spec_res.index_map_path) as s_src:
                                f_arr = f_src.read(1)
                                s_arr = s_src.read(1)
                                mh = min(f_arr.shape[0], s_arr.shape[0])
                                mw = min(f_arr.shape[1], s_arr.shape[1])
                                f_water = (f_arr[:mh, :mw] == 1)
                                s_water = (s_arr[:mh, :mw] > 0.0)
                                inter = np.sum(f_water & s_water)
                                uni = np.sum(f_water | s_water)
                                if uni > 0:
                                    agreement = float(inter) / float(uni)
                                else:
                                    agreement = 1.0
                        else:
                            pct_diff = abs(fusion_res.water_percentage - getattr(spec_res, "positive_percentage", 0.0))
                            agreement = max(0.0, 1.0 - (pct_diff / 100.0))
                    except Exception:
                        pct_diff = abs(fusion_res.water_percentage - getattr(spec_res, "positive_percentage", 0.0))
                        agreement = max(0.0, 1.0 - (pct_diff / 100.0))

                    agreement = round(agreement, 4)
                    computed_metrics["cross_tool_ndwi_agreement"] = agreement
                    conf_breakdown["cross_tool_ndwi_agreement"] = agreement

                    if agreement < 0.50:
                        warning_msg = f"Discrepancy Warning: Low spatial agreement ({round(agreement*100, 1)}%) between Optical-SAR Fusion water mask and NDWI index. Potential specular anomaly, shadow, or shallow water turbidity."
                        trace_logger.trace.actions_taken.append(warning_msg)
                        final_answer += f"\n\n⚠️ {warning_msg}"
                        base_confidence = max(0.40, round(0.90 - 0.35 * (1.0 - agreement), 4))
                        conf_breakdown["fusion_segmenter"] = base_confidence
                    else:
                        final_answer += f"\n• Spectral Cross-Check: NDWI spatial agreement score is {round(agreement*100, 1)}%."
                        base_confidence = round(0.60 * 0.92 + 0.40 * agreement, 4)
                        conf_breakdown["fusion_segmenter"] = base_confidence

            # Individual GUI layers for toggling
            if len(images) >= 2:
                opt_img = images[0] if images[0].modality != Modality.SAR else images[1]
                sar_img = images[1] if images[0].modality != Modality.SAR else images[0]
                if opt_img.bounds:
                    ob = opt_img.bounds
                    layers.append(LayerItem(
                        layer_id=f"optical_{uuid.uuid4().hex[:6]}",
                        name="Optical Imagery Reference",
                        type="footprint",
                        geojson=GeoJSONFeatureCollection(
                            type="FeatureCollection",
                            features=[GeoJSONFeature(
                                type="Feature",
                                geometry=GeoJSONGeometry(
                                    type="Polygon",
                                    coordinates=[[[ob.minx, ob.miny], [ob.maxx, ob.miny], [ob.maxx, ob.maxy], [ob.minx, ob.maxy], [ob.minx, ob.miny]]]
                                ),
                                properties={"modality": "optical", "name": opt_img.image_id}
                            )]
                        ),
                        color="#38BDF8",
                        opacity=0.4
                    ))
                if sar_img.bounds:
                    sb = sar_img.bounds
                    layers.append(LayerItem(
                        layer_id=f"sar_{uuid.uuid4().hex[:6]}",
                        name="SAR Radar Reference",
                        type="footprint",
                        geojson=GeoJSONFeatureCollection(
                            type="FeatureCollection",
                            features=[GeoJSONFeature(
                                type="Feature",
                                geometry=GeoJSONGeometry(
                                    type="Polygon",
                                    coordinates=[[[sb.minx, sb.miny], [sb.maxx, sb.miny], [sb.maxx, sb.maxy], [sb.minx, sb.maxy], [sb.minx, sb.miny]]]
                                ),
                                properties={"modality": "sar", "name": sar_img.image_id}
                            )]
                        ),
                        color="#A855F7",
                        opacity=0.4
                    ))

            if fusion_res.geojson and fusion_res.geojson.features:
                water_feats = [f for f in fusion_res.geojson.features if f.properties.get("class") == "water"]
                built_feats = [f for f in fusion_res.geojson.features if f.properties.get("class") == "built_up"]

                if water_feats:
                    layers.append(LayerItem(
                        layer_id=f"water_mask_{uuid.uuid4().hex[:6]}",
                        name="Water Mask",
                        type="geojson",
                        geojson=GeoJSONFeatureCollection(type="FeatureCollection", features=water_feats),
                        color="#00E5FF",
                        opacity=0.80
                    ))

                if built_feats:
                    layers.append(LayerItem(
                        layer_id=f"built_up_mask_{uuid.uuid4().hex[:6]}",
                        name="Built-up Mask",
                        type="geojson",
                        geojson=GeoJSONFeatureCollection(type="FeatureCollection", features=built_feats),
                        color="#F59E0B",
                        opacity=0.80
                    ))



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

        # 9. Multi-factor Confidence Scoring (backend/confidence/scorer.py)
        conf_eval = self.confidence_scorer.compute_confidence(
            raw_model_confidence=base_confidence,
            tool_name=list(tool_results.keys())[0] if tool_results else None,
            cross_tool_agreement=computed_metrics.get("cross_tool_ndwi_agreement"),
            actions_taken=trace_logger.trace.actions_taken,
            computed_metrics=computed_metrics,
            input_validation=val_record
        )
        final_confidence = conf_eval["confidence_score"]
        conf_breakdown.update(conf_eval["confidence_breakdown"])

        synth_ms = (time.time() - t_synth_0) * 1000.0
        trace_logger.log_stage_latency("synthesis_ms", synth_ms)

        # 10. Finalize Trace
        trace = trace_logger.finalize(
            computed_metrics=computed_metrics,
            confidence_score=final_confidence,
            confidence_breakdown=conf_breakdown,
            fallback_used=fallback_used
        )

        response = QueryResponse(
            session_id=session_id,
            query=request.query,
            task=task,
            answer=final_answer,
            layers=layers,
            computed_metrics=computed_metrics,
            confidence_score=round(final_confidence, 4),
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
