import re
from typing import List, Tuple, Dict, Any, Optional
from ..app.schemas.common import Modality, TaskType, ImageMetadata

class IncompatibleInputError(ValueError):
    """Raised when query intent and image inputs are fundamentally incompatible."""
    pass

class RuleBasedRouter:
    """
    Deterministic rule-based router (Controller v0) mapping query keywords + input configuration
    to one of the 6 canonical task types and generating the ordered tool sequence.
    Rejects incompatible inputs with clear human-readable error messages.
    """
    GROUNDING_KEYWORDS = [
        "where", "locate", "detect", "ground", "find", "boxes", "box",
        "show me", "identify the position", "coordinates", "highlight",
        "bounding", "pinpoint", "localize"
    ]
    CAPTION_KEYWORDS = [
        "describe", "caption", "overview", "scene summary", "what is this scene",
        "summarize the image", "tell me about this area", "give a summary", "scene description"
    ]
    CHANGE_KEYWORDS = [
        "change", "difference", "between", "decrease", "increase", "expanded",
        "shrink", "loss", "growth", "temporal", "compare", "shift", "deforestation",
        "what changed", "before and after", "evolution"
    ]
    CHANGE_MAP_KEYWORDS = [
        "change map", "mask", "area changed", "how much changed", "percentage changed",
        "spatial extent of change", "delineate change", "quantify change"
    ]
    FUSION_KEYWORDS = [
        "fuse", "fusion", "optical and sar", "sar and optical", "cross-modal",
        "radar and optical", "water extraction with sar", "all-weather", "penetrate clouds"
    ]
    SPECTRAL_KEYWORDS = [
        "ndvi", "ndwi", "ndbi", "vegetation index", "water index", "spectral"
    ]

    def classify_task(
        self,
        query: str,
        images: List[ImageMetadata],
        task_override: Optional[TaskType] = None
    ) -> TaskType:
        """
        Classifies task type from query keywords and input image configuration.
        Rejects incompatible requests with clear error messages.
        """
        if task_override is not None:
            return task_override

        q = query.lower().strip()
        img_count = len(images)
        modalities = [img.modality for img in images]

        # Incompatibility checks based on query intent vs input count
        has_change_intent = any(k in q for k in self.CHANGE_KEYWORDS) or any(k in q for k in self.CHANGE_MAP_KEYWORDS)
        has_fusion_intent = any(k in q for k in self.FUSION_KEYWORDS)

        # 1. Reject single image for change query
        if has_change_intent and img_count < 2:
            raise IncompatibleInputError(
                f"Incompatible input: Query specifies change detection ('{query}'), "
                f"which requires exactly 2 images (bi-temporal pair: pre-event and post-event), "
                f"but only {img_count} image was provided."
            )

        # 2. Reject single image for optical-SAR fusion query
        if has_fusion_intent and img_count < 2:
            raise IncompatibleInputError(
                f"Incompatible input: Query specifies Optical-SAR fusion ('{query}'), "
                f"which requires exactly 2 images (one Optical/Multispectral and one SAR), "
                f"but only {img_count} image was provided."
            )

        # 3. Reject dual image when query is explicitly single-image caption/grounding without change intent
        if img_count > 1 and not (has_change_intent or has_fusion_intent):
            has_grounding = any(re.search(rf"\b{k}\b", q) for k in self.GROUNDING_KEYWORDS)
            has_caption = any(re.search(rf"\b{k}\b", q) for k in self.CAPTION_KEYWORDS)
            # If 2 optical images provided without change keywords
            if (has_grounding or has_caption) and not (Modality.SAR in modalities):
                raise IncompatibleInputError(
                    f"Incompatible input: Single-image analysis task requested ('{query}') "
                    f"requires exactly 1 image, but received {img_count} images without bi-temporal or fusion context."
                )

        # Task assignment logic
        # Dual Image: Optical + SAR Pair or Cross-Modal query
        if img_count == 2:
            if (Modality.SAR in modalities) or any(k in q for k in self.FUSION_KEYWORDS) or (("water" in q or "reservoir" in q) and ("built" in q or "urban" in q)):
                return TaskType.OPTICAL_SAR_ANALYSIS

        # Dual Image: Bi-temporal Pair
        if img_count == 2:
            if any(k in q for k in self.CHANGE_MAP_KEYWORDS):
                return TaskType.CHANGE_MAP
            return TaskType.CHANGE_VQA

        # Single Image Tasks
        if any(re.search(rf"\b{k}\b", q) for k in self.GROUNDING_KEYWORDS):
            return TaskType.GROUNDING

        if any(re.search(rf"\b{k}\b", q) for k in self.CAPTION_KEYWORDS):
            return TaskType.CAPTION

        # Default single image task: Single VQA
        return TaskType.SINGLE_VQA

    def plan_tools(
        self,
        task: TaskType,
        query: str,
        images: List[ImageMetadata]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Builds the ordered sequence of tool calls with strictly whitelisted parameters.
        """
        plan: List[Tuple[str, Dict[str, Any]]] = []
        q = query.lower()

        if task == TaskType.SINGLE_VQA:
            img_path = images[0].file_path
            if any(k in q for k in self.SPECTRAL_KEYWORDS):
                idx_type = "NDWI" if "water" in q or "ndwi" in q else "NDVI"
                plan.append(("spectral_index", {
                    "image_path": img_path,
                    "index_type": idx_type,
                    "threshold": 0.2
                }))
            plan.append(("rs_vqa", {
                "query": query,
                "image_path": img_path,
                "confidence_threshold": 0.5
            }))

        elif task == TaskType.CAPTION:
            plan.append(("rs_caption", {
                "image_path": images[0].file_path,
                "max_length": 120,
                "style": "detailed"
            }))

        elif task == TaskType.GROUNDING:
            plan.append(("rs_grounding", {
                "query": query,
                "image_path": images[0].file_path,
                "box_threshold": 0.3,
                "text_threshold": 0.25
            }))

        elif task == TaskType.CHANGE_VQA:
            plan.append(("change_map", {
                "image_t1_path": images[0].file_path,
                "image_t2_path": images[1].file_path,
                "threshold": 0.45,
                "min_area_m2": 100.0
            }))
            plan.append(("change_vqa", {
                "query": query,
                "image_t1_path": images[0].file_path,
                "image_t2_path": images[1].file_path
            }))

        elif task == TaskType.CHANGE_MAP:
            plan.append(("change_map", {
                "image_t1_path": images[0].file_path,
                "image_t2_path": images[1].file_path,
                "threshold": 0.5,
                "min_area_m2": 100.0
            }))
            plan.append(("change_vqa", {
                "query": query,
                "image_t1_path": images[0].file_path,
                "image_t2_path": images[1].file_path
            }))

        elif task == TaskType.OPTICAL_SAR_ANALYSIS:
            opt_path = images[0].file_path if images[0].modality != Modality.SAR else images[1].file_path
            sar_path = images[1].file_path if images[0].modality != Modality.SAR else images[0].file_path

            plan.append(("optical_sar_fusion", {
                "optical_path": opt_path,
                "sar_path": sar_path,
                "target_classes": ["water", "built_up"],
                "confidence_threshold": 0.5
            }))
            plan.append(("spectral_index", {
                "image_path": opt_path,
                "index_type": "NDWI",
                "threshold": 0.15
            }))

        return plan
