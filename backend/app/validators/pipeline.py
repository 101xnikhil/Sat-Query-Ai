import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from ..schemas.common import Modality, TaskType, ImageMetadata
from ..schemas.trace import ValidationRecord
from ..config import Settings
from .spatial import compute_extent_overlap, reproject_raster_to_match
from .sar import preprocess_sar_image
from ..ingestion.reader import read_image_metadata

class ValidationError(Exception):
    """Custom exception raised when inputs violate validator constraints."""
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or [message]

def validate_task_inputs(
    task: TaskType,
    images: List[ImageMetadata],
    settings: Settings,
    output_dir: Optional[str] = None
) -> Tuple[bool, ValidationRecord, List[ImageMetadata]]:
    """
    Validates input count, modality compatibility, spatial overlap, CRS reprojection,
    and executes SAR preprocessing.
    """
    record = ValidationRecord(passed=True)
    out_dir = Path(output_dir or settings.app.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_images = [img.model_copy() for img in images]

    # 1. Image count check
    single_image_tasks = [TaskType.SINGLE_VQA, TaskType.CAPTION, TaskType.GROUNDING]
    dual_image_tasks = [TaskType.CHANGE_VQA, TaskType.CHANGE_MAP, TaskType.OPTICAL_SAR_ANALYSIS]

    if task in single_image_tasks:
        if len(images) != 1:
            err = f"Task '{task.value}' requires exactly 1 image, but received {len(images)}."
            record.errors.append(err)
            record.passed = False
            return False, record, processed_images

    elif task in dual_image_tasks:
        if len(images) != 2:
            err = f"Task '{task.value}' requires exactly 2 images, but received {len(images)}."
            record.errors.append(err)
            record.passed = False
            return False, record, processed_images

    # 2. Modality Checks
    if task == TaskType.OPTICAL_SAR_ANALYSIS:
        modalities = [img.modality for img in images]
        has_sar = Modality.SAR in modalities
        has_optical = any(m in [Modality.OPTICAL, Modality.MULTISPECTRAL] for m in modalities)
        if not (has_sar and has_optical):
            err = (
                f"Task '{task.value}' requires one Optical/Multispectral image and one SAR image. "
                f"Provided modalities: {[m.value for m in modalities]}."
            )
            record.errors.append(err)
            record.passed = False
            return False, record, processed_images

    elif task in [TaskType.CHANGE_VQA, TaskType.CHANGE_MAP]:
        # Temporal checks
        if images[0].acquisition_date and images[1].acquisition_date:
            try:
                d1 = datetime.fromisoformat(images[0].acquisition_date.replace("Z", "+00:00"))
                d2 = datetime.fromisoformat(images[1].acquisition_date.replace("Z", "+00:00"))
                if d1 == d2:
                    record.warnings.append("Bi-temporal images have identical acquisition dates.")
                elif d1 > d2:
                    # Swap so images are strictly chronological (t1 earlier, t2 later)
                    processed_images = [images[1].model_copy(), images[0].model_copy()]
                    record.warnings.append("Images reordered chronologically: t1 is earlier than t2.")
                    record.actions_taken.append("Reordered bi-temporal images chronologically: image 2 set as T1, image 1 set as T2.")
            except Exception:
                record.warnings.append("Could not parse acquisition dates for temporal ordering.")

    # 3. Spatial Validation for multi-image tasks
    if len(processed_images) == 2:
        img1, img2 = processed_images[0], processed_images[1]
        record.crs_check = {
            "img1_crs": img1.crs,
            "img2_crs": img2.crs,
            "crs_match": (img1.crs == img2.crs) if (img1.crs and img2.crs) else True
        }

        # Check overlap
        iou, overlap_meta = compute_extent_overlap(img1, img2)
        record.overlap_iou = iou

        if iou < settings.validation.min_overlap_iou:
            err = (
                f"Spatial extent overlap (IoU = {iou:.3f}) is below minimum threshold "
                f"{settings.validation.min_overlap_iou:.2f}. Images do not sufficiently overlap."
            )
            record.errors.append(err)
            record.passed = False
            return False, record, processed_images

        # CRS Reprojection if mismatched
        if img1.crs and img2.crs and img1.crs != img2.crs and settings.validation.auto_reproject:
            reprojected_path = str(out_dir / f"reprojected_{img2.image_id}.tif")
            reproj_info = reproject_raster_to_match(
                source_path=img2.file_path,
                reference_path=img1.file_path,
                output_path=reprojected_path
            )
            record.reprojections.append(reproj_info)
            record.actions_taken.append(f"Auto-reprojected {img2.image_id} to common grid matching {img1.image_id} ({img1.crs}).")
            # Update metadata of reprojected img2
            updated_meta = read_image_metadata(
                file_path=reprojected_path,
                image_id=f"reprojected_{img2.image_id}",
                modality_hint=img2.modality,
                acquisition_date=img2.acquisition_date
            )
            processed_images[1] = updated_meta

    # 4. SAR Preprocessing
    for idx, img in enumerate(processed_images):
        if img.modality == Modality.SAR:
            preprocessed_path = str(out_dir / f"sar_preprocessed_{img.image_id}.tif")
            sar_info = preprocess_sar_image(
                input_path=img.file_path,
                output_path=preprocessed_path,
                db_conversion=settings.validation.sar.db_conversion,
                apply_filter=bool(settings.validation.sar.speckle_filter == "lee"),
                window_size=settings.validation.sar.filter_window_size,
                epsilon=settings.validation.sar.epsilon
            )
            record.sar_preprocessing.append(sar_info)
            record.actions_taken.append(f"Preprocessed SAR image {img.image_id}: linear to dB conversion and Lee speckle filtering.")
            updated_meta = read_image_metadata(
                file_path=preprocessed_path,
                image_id=f"preprocessed_{img.image_id}",
                modality_hint=Modality.SAR,
                acquisition_date=img.acquisition_date
            )
            processed_images[idx] = updated_meta

    return record.passed, record, processed_images
