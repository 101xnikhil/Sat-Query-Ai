from typing import Tuple, Optional, List
import numpy as np
from PIL import Image

from backend.app.schemas.common import Modality
from backend.ingest.reader import read_raster_array, read_image_metadata

def percentile_stretch(arr: np.ndarray, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
    """
    Applies empirical 2nd to 98th percentile contrast stretching,
    mapping values to uint8 [0, 255].
    """
    clean_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    val_min = float(np.percentile(clean_arr, p_min))
    val_max = float(np.percentile(clean_arr, p_max))

    if val_max <= val_min:
        val_min = float(np.min(clean_arr))
        val_max = float(np.max(clean_arr))

    if val_max > val_min:
        stretched = (clean_arr - val_min) / (val_max - val_min)
        stretched = np.clip(stretched * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        stretched = np.zeros_like(clean_arr, dtype=np.uint8)

    return stretched

def render_multispectral_rgb(arr: np.ndarray, band_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, str]:
    """
    Extracts RGB channels from multispectral raster (C, H, W) and applies 2nd-98th percentile stretch.
    Default band indices: [2, 1, 0] for BGR -> RGB or [0, 1, 2].
    Returns (H, W, 3) uint8 numpy array and rendering descriptor.
    """
    if arr.ndim == 2:
        stretched = percentile_stretch(arr)
        rgb = np.stack([stretched, stretched, stretched], axis=-1)
        return rgb, "panchromatic_grayscale_stretch"

    bands = arr.shape[0]
    if bands >= 3:
        if band_indices and len(band_indices) >= 3:
            r = percentile_stretch(arr[band_indices[0]])
            g = percentile_stretch(arr[band_indices[1]])
            b = percentile_stretch(arr[band_indices[2]])
        else:
            # If 4 bands (e.g. B, G, R, NIR), Red=2, Green=1, Blue=0
            r = percentile_stretch(arr[2] if bands >= 3 else arr[0])
            g = percentile_stretch(arr[1] if bands >= 2 else arr[0])
            b = percentile_stretch(arr[0])
        rgb = np.stack([r, g, b], axis=-1)
    else:
        gray = percentile_stretch(arr[0])
        rgb = np.stack([gray, gray, gray], axis=-1)

    return rgb, "multispectral_true_color_rgb"

def render_false_color_cir(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Renders Color-Infrared (CIR) composite (NIR -> Red, Red -> Green, Green -> Blue).
    Highlights active vegetation in red and water in dark tones.
    Returns (H, W, 3) uint8 numpy array and rendering descriptor.
    """
    if arr.ndim == 2 or arr.shape[0] < 4:
        # Fallback to high-contrast stretch if NIR band not present
        return render_multispectral_rgb(arr)

    # Standard Sentinel-2 / Planet CIR: Band 4 (NIR) -> R, Band 3 (Red) -> G, Band 2 (Green) -> B
    nir = percentile_stretch(arr[3])
    red = percentile_stretch(arr[2])
    green = percentile_stretch(arr[1])

    cir = np.stack([nir, red, green], axis=-1)
    return cir, "multispectral_false_color_cir"

def render_sar_db(
    arr: np.ndarray,
    db_conversion: bool = True,
    pseudo_color: bool = False
) -> Tuple[np.ndarray, str]:
    """
    Converts SAR linear amplitude to dB scale and applies percentile contrast stretching.
    Returns (H, W, 3) uint8 numpy array and rendering descriptor.
    """
    slice_2d = arr[0] if arr.ndim == 3 else arr

    if db_conversion:
        # Check if already in dB (has negative values)
        if np.min(slice_2d) >= 0.0:
            sar_db = 10.0 * np.log10(np.clip(slice_2d.astype(np.float32), 1e-7, None))
        else:
            sar_db = slice_2d.astype(np.float32)
    else:
        sar_db = slice_2d.astype(np.float32)

    gray = percentile_stretch(sar_db, p_min=1.0, p_max=99.0)

    if pseudo_color:
        # Simple high-contrast pseudo-color mapping (dark blue -> cyan -> yellow)
        r = gray
        g = np.clip(gray * 1.2, 0, 255).astype(np.uint8)
        b = 255 - gray
        sar_img = np.stack([r, g, b], axis=-1)
        mode = "sar_db_pseudocolor"
    else:
        sar_img = np.stack([gray, gray, gray], axis=-1)
        mode = "sar_db_grayscale"

    return sar_img, mode

def auto_render_for_vlm(
    raster_path: str,
    modality: Optional[Modality] = None,
    query: str = ""
) -> Tuple[np.ndarray, str]:
    """
    Automatically selects optimal remote sensing rendering composite based on
    modality, available bands, and natural language query keywords.
    """
    arr = read_raster_array(raster_path)
    meta = read_image_metadata(raster_path)
    active_modality = modality or meta.modality
    q = query.lower()

    if active_modality == Modality.SAR:
        pseudo = ("pseudo" in q or "water" in q or "flood" in q)
        return render_sar_db(arr, db_conversion=True, pseudo_color=pseudo)

    # If query is vegetation/crop/forestry/water focused and raster has >= 4 bands
    if arr.ndim == 3 and arr.shape[0] >= 4:
        if any(k in q for k in ["vegetation", "crop", "forest", "tree", "plant", "cir", "false color"]):
            return render_false_color_cir(arr)

    return render_multispectral_rgb(arr)
