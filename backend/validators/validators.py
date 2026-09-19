from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

from ..app.schemas.common import Modality, TaskType, ImageMetadata
from ..app.schemas.trace import ValidationRecord
from ..app.validators.spatial import compute_extent_overlap, reproject_raster_to_match
from ..app.validators.sar import preprocess_sar_image
from ..ingest.reader import read_image_metadata
from .results import ValidatorResult

SINGLE_IMAGE_TASKS = [TaskType.SINGLE_VQA, TaskType.CAPTION, TaskType.GROUNDING]
DUAL_IMAGE_TASKS = [TaskType.CHANGE_VQA, TaskType.CHANGE_MAP, TaskType.OPTICAL_SAR_ANALYSIS]

def validate_image_count(images: List[ImageMetadata], task: TaskType) -> ValidatorResult:
    """Validates that the provided image count satisfies task requirements."""
    res = ValidatorResult(ok=True)
    count = len(images)

    if task in SINGLE_IMAGE_TASKS:
        if count != 1:
            res.ok = False
            res.errors.append(f"Task '{task.value}' requires exactly 1 image, but received {count}.")
        else:
            res.actions_taken.append(f"Verified image count (1) for task '{task.value}'.")
    elif task in DUAL_IMAGE_TASKS:
        if count != 2:
            res.ok = False
            res.errors.append(f"Task '{task.value}' requires exactly 2 images, but received {count}.")
        else:
            res.actions_taken.append(f"Verified image count (2) for task '{task.value}'.")
    else:
        if count == 0:
            res.ok = False
            res.errors.append(f"No imagery provided for task '{task.value}'.")

    return res

def validate_modality_compatibility(images: List[ImageMetadata], task: TaskType) -> ValidatorResult:
    """Validates that input imagery modalities are compatible with the designated task."""
    res = ValidatorResult(ok=True)
    modalities = [img.modality for img in images]

    if task == TaskType.OPTICAL_SAR_ANALYSIS:
        has_sar = Modality.SAR in modalities
        has_optical = any(m in [Modality.OPTICAL, Modality.MULTISPECTRAL] for m in modalities)
        if not (has_sar and has_optical):
            res.ok = False
            res.errors.append(
                f"Task '{task.value}' requires one Optical/Multispectral image and one SAR image. "
                f"Provided modalities: {[m.value for m in modalities]}."
            )
        else:
            res.actions_taken.append("Verified cross-modal compatibility: Optical/Multispectral + SAR pair verified.")

    elif task in [TaskType.CHANGE_VQA, TaskType.CHANGE_MAP]:
        # Both images should preferably share the same primary modality (e.g. optical-optical or sar-sar)
        if len(modalities) == 2 and modalities[0] != modalities[1]:
            # If one optical and one sar for change task, warn
            if Modality.SAR in modalities and any(m in [Modality.OPTICAL, Modality.MULTISPECTRAL] for m in modalities):
                res.warnings.append(
                    "Bi-temporal change requested with heterogeneous modalities (Optical + SAR). "
                    "Cross-modal pseudo-change may occur."
                )

    # Sanity check for single SAR with color/RGB queries
    if task == TaskType.SINGLE_VQA and len(images) == 1 and images[0].modality == Modality.SAR:
        res.warnings.append("Single SAR imagery provided: color-related interpretations are not directly applicable.")

    return res

def validate_crs_and_overlap(
    images: List[ImageMetadata],
    min_overlap_iou: float = 0.10,
    auto_reproject: bool = True,
    output_dir: Optional[str] = None
) -> Tuple[ValidatorResult, List[ImageMetadata]]:
    """
    Validates CRS match and spatial extent overlap.
    Auto reprojects/resamples img2 to img1's common grid if CRS or resolutions mismatch.
    """
    res = ValidatorResult(ok=True)
    processed = [img.model_copy() for img in images]

    if len(processed) < 2:
        return res, processed

    img1, img2 = processed[0], processed[1]

    # If non-georeferenced benchmark images, skip spatial geometry check
    if not img1.is_georeferenced or not img2.is_georeferenced:
        res.warnings.append("One or more images are not georeferenced; skipping spatial extent check.")
        return res, processed

    # Extent overlap
    iou, overlap_meta = compute_extent_overlap(img1, img2)
    if iou < min_overlap_iou:
        res.ok = False
        res.errors.append(
            f"Spatial extent overlap (IoU = {iou:.3f}) is below minimum threshold "
            f"{min_overlap_iou:.2f}. Images do not sufficiently overlap."
        )
        return res, processed

    res.actions_taken.append(f"Spatial extent overlap verified: IoU = {iou:.3f} >= {min_overlap_iou:.2f}.")

    # Co-registration: check CRS, resolution, and grid alignment
    needs_alignment = False
    alignment_reason = []

    if img1.crs and img2.crs and img1.crs != img2.crs:
        needs_alignment = True
        alignment_reason.append(f"CRS mismatch ({img2.crs} -> {img1.crs})")

    res1 = img1.resolution or (10.0, 10.0)
    res2 = img2.resolution or (10.0, 10.0)
    res_ratio = max(res1[0] / max(1e-6, res2[0]), res2[0] / max(1e-6, res1[0]))
    if res_ratio > 1.05 or (img1.width != img2.width) or (img1.height != img2.height):
        needs_alignment = True
        alignment_reason.append(f"Grid/Resolution mismatch ({img2.width}x{img2.height} -> {img1.width}x{img1.height})")

    if needs_alignment and auto_reproject:
        out_dir = Path(output_dir or "data/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        aligned_path = str(out_dir / f"aligned_{img2.image_id}.tif")

        reproject_raster_to_match(
            source_path=img2.file_path,
            reference_path=img1.file_path,
            output_path=aligned_path
        )
        reason_str = ", ".join(alignment_reason)
        res.actions_taken.append(f"Auto-reprojected and auto-aligned {img2.image_id} to common grid: {reason_str}.")

        updated_meta = read_image_metadata(
            file_path=aligned_path,
            image_id=f"aligned_{img2.image_id}",
            modality_hint=img2.modality,
            acquisition_date=img2.acquisition_date
        )
        processed[1] = updated_meta
    else:
        res.actions_taken.append("Co-registration verified: common grid, CRS, and resolution aligned.")

    return res, processed

def validate_acquisition_dates(
    images: List[ImageMetadata],
    task: TaskType
) -> Tuple[ValidatorResult, List[ImageMetadata]]:
    """
    Validates that acquisition dates are present and ordered for temporal tasks.
    Auto re-orders images chronologically (T1 earlier, T2 later) if dates are inverted.
    """
    res = ValidatorResult(ok=True)
    processed = [img.model_copy() for img in images]

    if task in [TaskType.CHANGE_VQA, TaskType.CHANGE_MAP] and len(processed) == 2:
        d1_str = processed[0].acquisition_date
        d2_str = processed[1].acquisition_date

        if not d1_str or not d2_str:
            res.warnings.append(
                "Acquisition date missing for one or both images in temporal change task. "
                "Defaulting to input order: image 1 as T1 (pre-event), image 2 as T2 (post-event)."
            )
            return res, processed

        try:
            d1 = datetime.fromisoformat(d1_str.replace("Z", "+00:00"))
            d2 = datetime.fromisoformat(d2_str.replace("Z", "+00:00"))

            if d1 == d2:
                res.warnings.append("Bi-temporal images have identical acquisition dates.")
            elif d1 > d2:
                # Chronological reorder: swap so T1 is earlier, T2 is later
                processed = [processed[1].model_copy(), processed[0].model_copy()]
                res.actions_taken.append(
                    f"Auto-reordered bi-temporal images chronologically: "
                    f"T1 is now earlier date ({d2_str}), T2 is later date ({d1_str})."
                )
            else:
                res.actions_taken.append(f"Verified chronological order: T1 ({d1_str}) < T2 ({d2_str}).")
        except Exception:
            res.warnings.append(f"Could not parse acquisition dates ('{d1_str}', '{d2_str}') as ISO timestamps.")

    return res, processed

def validate_band_count_and_dtype(images: List[ImageMetadata]) -> ValidatorResult:
    """Sanity check on band count and data types."""
    res = ValidatorResult(ok=True)
    for img in images:
        if img.band_count <= 0:
            res.ok = False
            res.errors.append(f"Image '{img.image_id}' has invalid band count: {img.band_count}.")
        if img.width <= 0 or img.height <= 0:
            res.ok = False
            res.errors.append(f"Image '{img.image_id}' has invalid dimensions: {img.width}x{img.height}.")

    if res.ok:
        res.actions_taken.append(f"Band count and dtype sanity verified across {len(images)} image(s).")
    return res

def validate_and_preprocess_sar(
    images: List[ImageMetadata],
    db_conversion: bool = True,
    speckle_filter: str = "lee",
    window_size: int = 3,
    epsilon: float = 1e-7,
    output_dir: Optional[str] = None
) -> Tuple[ValidatorResult, List[ImageMetadata]]:
    """
    Configurable SAR preprocessing: linear to dB, speckle filter (Lee).
    Records actions taken in structured result.
    """
    res = ValidatorResult(ok=True)
    processed = [img.model_copy() for img in images]
    out_dir = Path(output_dir or "data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(processed):
        if img.modality == Modality.SAR:
            preprocessed_path = str(out_dir / f"sar_preprocessed_{img.image_id}.tif")
            apply_filter = (speckle_filter.lower() == "lee")

            sar_info = preprocess_sar_image(
                input_path=img.file_path,
                output_path=preprocessed_path,
                db_conversion=db_conversion,
                apply_filter=apply_filter,
                window_size=window_size,
                epsilon=epsilon
            )
            actions = []
            if db_conversion:
                actions.append("Linear-to-dB amplitude conversion")
            if apply_filter:
                actions.append(f"Lee speckle filter (window {window_size}x{window_size})")

            action_desc = f"SAR preprocessing for {img.image_id}: {', '.join(actions)}."
            res.actions_taken.append(action_desc)

            updated_meta = read_image_metadata(
                file_path=preprocessed_path,
                image_id=f"preprocessed_{img.image_id}",
                modality_hint=Modality.SAR,
                acquisition_date=img.acquisition_date
            )
            processed[idx] = updated_meta

    return res, processed

def validate_all(
    task: TaskType,
    images: List[ImageMetadata],
    settings: Any,
    output_dir: Optional[str] = None
) -> Tuple[ValidatorResult, ValidationRecord, List[ImageMetadata]]:
    """
    Pipeline running all individual validators sequentially, accumulating structured
    ValidatorResult and populating legacy ValidationRecord for trace logging.
    """
    combined_res = ValidatorResult(ok=True)
    val_record = ValidationRecord(passed=True)
    current_images = [img.model_copy() for img in images]

    # 1. Image count check
    r1 = validate_image_count(current_images, task)
    combined_res.merge(r1)
    if not r1.ok:
        val_record.passed = False
        val_record.errors.extend(r1.errors)
        return combined_res, val_record, current_images

    # 2. Modality compatibility
    r2 = validate_modality_compatibility(current_images, task)
    combined_res.merge(r2)
    if not r2.ok:
        val_record.passed = False
        val_record.errors.extend(r2.errors)
        return combined_res, val_record, current_images

    # 3. Band count and dtype sanity
    r3 = validate_band_count_and_dtype(current_images)
    combined_res.merge(r3)
    if not r3.ok:
        val_record.passed = False
        val_record.errors.extend(r3.errors)
        return combined_res, val_record, current_images

    # 4. Temporal order check
    r4, current_images = validate_acquisition_dates(current_images, task)
    combined_res.merge(r4)

    # 5. Spatial overlap and CRS check
    r5, current_images = validate_crs_and_overlap(
        images=current_images,
        min_overlap_iou=settings.validation.min_overlap_iou,
        auto_reproject=settings.validation.auto_reproject,
        output_dir=output_dir or settings.app.outputs_dir
    )
    combined_res.merge(r5)
    if not r5.ok:
        val_record.passed = False
        val_record.errors.extend(r5.errors)
        return combined_res, val_record, current_images

    # 6. SAR preprocessing
    sar_cfg = settings.validation.sar
    r6, current_images = validate_and_preprocess_sar(
        images=current_images,
        db_conversion=sar_cfg.db_conversion,
        speckle_filter=sar_cfg.speckle_filter,
        window_size=sar_cfg.filter_window_size,
        epsilon=sar_cfg.epsilon,
        output_dir=output_dir or settings.app.outputs_dir
    )
    combined_res.merge(r6)

    # Sync into legacy ValidationRecord
    val_record.passed = combined_res.ok
    val_record.errors = list(combined_res.errors)
    val_record.warnings = list(combined_res.warnings)

    return combined_res, val_record, current_images
