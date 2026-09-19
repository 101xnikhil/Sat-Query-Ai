import re
from typing import List, Tuple, Dict, Any, Optional
from ..schemas.common import Modality, TaskType, ImageMetadata

class RuleBasedRouter:
    """
    Deterministic rule-based router that classifies user query and input configuration
    into one of the 6 canonical task types and determines the ordered tool execution plan.
    """
    GROUNDING_KEYWORDS = [
        "where", "locate", "detect", "ground", "find", "boxes", "box",
        "show me", "identify the position", "coordinates", "highlight"
    ]
    CAPTION_KEYWORDS = [
        "describe", "caption", "overview", "scene summary", "what is this scene",
        "summarize the image", "tell me about this area"
    ]
    CHANGE_KEYWORDS = [
        "change", "difference", "between", "decrease", "increase", "expanded",
        "shrink", "loss", "growth", "temporal", "compare", "shift", "deforestation"
    ]
    CHANGE_MAP_KEYWORDS = [
        "change map", "mask", "area changed", "how much changed", "percentage changed",
        "spatial extent of change"
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
        if task_override is not None:
            return task_override

        q = query.lower()
        img_count = len(images)
        modalities = [img.modality for img in images]

        # 1. Dual Image: Optical + SAR Pair
        if img_count == 2 and (Modality.SAR in modalities) and any(m in [Modality.OPTICAL, Modality.MULTISPECTRAL] for m in modalities):
            return TaskType.OPTICAL_SAR_ANALYSIS

        # 2. Dual Image: Bi-temporal Pair
        if img_count == 2:
            if any(k in q for k in self.CHANGE_MAP_KEYWORDS):
                return TaskType.CHANGE_MAP
            return TaskType.CHANGE_VQA

        # 3. Single Image Tasks
        if any(re.search(rf"\b{k}\b", q) for k in self.GROUNDING_KEYWORDS):
            return TaskType.GROUNDING

        if any(re.search(rf"\b{k}\b", q) for k in self.CAPTION_KEYWORDS):
            return TaskType.CAPTION

        # Default single image task
        return TaskType.SINGLE_VQA

    def plan_tools(
        self,
        task: TaskType,
        query: str,
        images: List[ImageMetadata]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Builds the sequence of tools to run with strictly whitelisted parameters.
        """
        plan: List[Tuple[str, Dict[str, Any]]] = []
        q = query.lower()

        if task == TaskType.SINGLE_VQA:
            img_path = images[0].file_path
            # If query mentions spectral indices (NDVI/NDWI), chain spectral_index first as evidence cross-check!
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
            plan.append(("change_vqa", {
                "query": query,
                "image_t1_path": images[0].file_path,
                "image_t2_path": images[1].file_path
            }))
            # Add change_map to ground numerical stats
            plan.append(("change_map", {
                "image_t1_path": images[0].file_path,
                "image_t2_path": images[1].file_path,
                "threshold": 0.45,
                "min_area_m2": 100.0
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
            # Sort optical vs SAR
            opt_path = images[0].file_path if images[0].modality != Modality.SAR else images[1].file_path
            sar_path = images[1].file_path if images[0].modality != Modality.SAR else images[0].file_path

            plan.append(("optical_sar_fusion", {
                "optical_path": opt_path,
                "sar_path": sar_path,
                "target_classes": ["water", "built_up"],
                "confidence_threshold": 0.5
            }))
            # Cross-check water with spectral_index
            plan.append(("spectral_index", {
                "image_path": opt_path,
                "index_type": "NDWI",
                "threshold": 0.15
            }))

        return plan
