from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import numpy as np

class ConfidenceScorer:
    """
    Config-driven multi-factor confidence calculator.
    Formula:
        Confidence = w_token * S_token + w_agreement * S_agreement + w_quality * S_quality
    Where:
        S_token: Calibrated probability / logits from VLM or vision models.
        S_agreement: Spatial and semantic consensus across complementary tools (e.g. Fusion vs NDWI).
        S_quality: 1.0 - sum(quality_penalties) based on CRS reprojection, resampling, missing bands, low overlap.
    """
    def __init__(self, config_path: Optional[str] = None):
        cfg_file = Path(config_path or "configs/confidence.yaml")
        if not cfg_file.exists():
            cfg_file = Path(__file__).resolve().parents[2] / "configs" / "confidence.yaml"

        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "weights": {
                    "token_probability": 0.40,
                    "cross_tool_agreement": 0.35,
                    "input_quality": 0.25
                },
                "baselines": {
                    "vlm_default": 0.88,
                    "segmentation_default": 0.90,
                    "grounding_default": 0.85,
                    "caption_default": 0.88
                },
                "quality_penalties": {
                    "resampling_applied": 0.10,
                    "missing_spectral_bands": 0.15,
                    "low_spatial_overlap": 0.20,
                    "sar_linear_scale_used": 0.05,
                    "single_band_degradation": 0.15,
                    "date_order_ambiguity": 0.08
                },
                "thresholds": {
                    "low_confidence_warning": 0.60,
                    "discrepancy_alert": 0.50
                }
            }

        self.weights = self.config.get("weights", {})
        self.penalties = self.config.get("quality_penalties", {})
        self.baselines = self.config.get("baselines", {})

    def compute_confidence(
        self,
        raw_model_confidence: Optional[float] = None,
        tool_name: Optional[str] = None,
        cross_tool_agreement: Optional[float] = None,
        actions_taken: Optional[List[str]] = None,
        computed_metrics: Optional[Dict[str, Any]] = None,
        input_validation: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Computes final confidence score and granular factor breakdown.
        """
        actions = actions_taken or []
        metrics = computed_metrics or {}

        # 1. Model Token Probability / Baseline Score
        if raw_model_confidence is not None and 0.0 <= raw_model_confidence <= 1.0:
            s_token = float(raw_model_confidence)
        else:
            # Fallback to tool baseline
            if tool_name in ("optical_sar_fusion", "change_map"):
                s_token = float(self.baselines.get("segmentation_default", 0.90))
            elif tool_name == "rs_grounding":
                s_token = float(self.baselines.get("grounding_default", 0.85))
            elif tool_name == "rs_caption":
                s_token = float(self.baselines.get("caption_default", 0.88))
            else:
                s_token = float(self.baselines.get("vlm_default", 0.88))

        # 2. Cross-Tool Agreement Score
        if cross_tool_agreement is not None:
            s_agreement = max(0.0, min(1.0, float(cross_tool_agreement)))
        elif "cross_tool_ndwi_agreement" in metrics:
            s_agreement = max(0.0, min(1.0, float(metrics["cross_tool_ndwi_agreement"])))
        elif "vlm_cross_check_agreed" in metrics:
            s_agreement = 1.0 if metrics["vlm_cross_check_agreed"] else 0.40
        else:
            # Nominal agreement when no multi-tool check was relevant
            s_agreement = 1.0

        # 3. Input Quality Penalties
        quality_deductions = 0.0
        breakdown_penalties: Dict[str, float] = {}

        # Check for resampling/reprojection
        resampling_flag = any("resample" in a.lower() or "reproject" in a.lower() for a in actions)
        if resampling_flag or "resampling_info" in metrics:
            p = float(self.penalties.get("resampling_applied", 0.10))
            quality_deductions += p
            breakdown_penalties["resampling_penalty"] = round(p, 4)

        # Check for missing spectral bands / single band degradation
        single_band_flag = any("single-band" in a.lower() or "missing band" in a.lower() or "not applicable" in a.lower() for a in actions)
        if single_band_flag or metrics.get("degradation_mode") == "single_band_degraded":
            p = float(self.penalties.get("single_band_degradation", 0.15))
            quality_deductions += p
            breakdown_penalties["missing_bands_penalty"] = round(p, 4)

        # Check for spatial overlap < 0.50
        iou = None
        if input_validation:
            iou = getattr(input_validation, "overlap_iou", None)
            if isinstance(input_validation, dict):
                iou = input_validation.get("overlap_iou")
        if iou is not None and iou < 0.50:
            p = float(self.penalties.get("low_spatial_overlap", 0.20))
            quality_deductions += p
            breakdown_penalties["low_overlap_penalty"] = round(p, 4)

        # Check date ordering warnings
        if any("date" in a.lower() and ("identical" in a.lower() or "missing" in a.lower()) for a in actions):
            p = float(self.penalties.get("date_order_ambiguity", 0.08))
            quality_deductions += p
            breakdown_penalties["date_order_penalty"] = round(p, 4)

        s_quality = max(0.20, min(1.0, 1.0 - quality_deductions))

        # 4. Weighted Linear Combination
        w_tok = float(self.weights.get("token_probability", 0.40))
        w_agr = float(self.weights.get("cross_tool_agreement", 0.35))
        w_qua = float(self.weights.get("input_quality", 0.25))

        # Normalize weights to guarantee sum == 1.0
        w_sum = w_tok + w_agr + w_qua
        w_tok /= w_sum
        w_agr /= w_sum
        w_qua /= w_sum

        final_score = (w_tok * s_token) + (w_agr * s_agreement) + (w_qua * s_quality)
        final_score = max(0.05, min(1.0, round(final_score, 4)))

        # 5. Build Comprehensive Breakdown
        breakdown = {
            "overall": final_score,
            "token_probability": round(s_token, 4),
            "cross_tool_agreement": round(s_agreement, 4),
            "input_quality": round(s_quality, 4),
            **breakdown_penalties
        }

        # Also add tool-specific aliases for backward compatibility with existing tests
        if tool_name:
            breakdown[tool_name] = round(final_score, 4)
        if "fusion_segmenter" in (tool_name or "") or "optical_sar_fusion" in (tool_name or ""):
            breakdown["fusion_segmenter"] = round(final_score, 4)
        if "cross_tool_ndwi_agreement" in metrics:
            breakdown["cross_tool_ndwi_agreement"] = round(s_agreement, 4)

        return {
            "confidence_score": final_score,
            "confidence_breakdown": breakdown
        }

_scorer: Optional[ConfidenceScorer] = None

def get_confidence_scorer(config_path: Optional[str] = None) -> ConfidenceScorer:
    global _scorer
    if _scorer is None or config_path is not None:
        _scorer = ConfidenceScorer(config_path)
    return _scorer
