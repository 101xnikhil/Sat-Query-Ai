import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from scipy.ndimage import uniform_filter
import rasterio

from ..schemas.common import ImageMetadata, Modality

def lee_speckle_filter(img: np.ndarray, window_size: int = 3, damping: float = 1.0) -> np.ndarray:
    """
    Applies the Lee filter for SAR speckle reduction.
    img: 2D numpy array (single band amplitude or intensity)
    window_size: odd integer (typically 3 or 5)
    """
    img_float = img.astype(np.float32)
    # Local mean
    mean = uniform_filter(img_float, size=window_size)
    # Local mean of square
    mean_sq = uniform_filter(img_float ** 2, size=window_size)
    # Local variance
    variance = mean_sq - mean ** 2
    variance = np.maximum(variance, 0.0)

    # Estimate noise variance (relative speckle variance ~ 1/looks, approx 0.25 for 4 looks or mean^2 / 4)
    overall_mean = np.mean(img_float)
    noise_var = (overall_mean ** 2) / 4.0

    # Weight factor
    weights = variance / (variance + noise_var + 1e-8)
    weights = np.clip(weights * damping, 0.0, 1.0)

    # Filtered image
    filtered = mean + weights * (img_float - mean)
    return filtered

def convert_to_db(img: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Converts linear amplitude / intensity to decibels (dB). Preserves if already in dB."""
    img_float = img.astype(np.float32)
    # If values contain negative numbers, it is already in decibels (e.g. -25 to +5 dB)
    if np.min(img_float) < 0.0 or (np.max(img_float) <= 30.0 and np.mean(img_float) < 5.0):
        return img_float
    linear = np.abs(img_float)
    db = 10.0 * np.log10(linear + epsilon)
    return db

def preprocess_sar_image(
    input_path: str,
    output_path: str,
    db_conversion: bool = True,
    apply_filter: bool = True,
    window_size: int = 3,
    epsilon: float = 1e-7
) -> Dict[str, Any]:
    """
    Performs end-to-end SAR preprocessing: dB conversion + speckle filtering.
    Preserves GeoTIFF projection and affine transform.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data = src.read()  # (C, H, W)
        
        # Check if already in dB
        is_already_db = np.min(data) < 0.0 or (np.max(data) <= 30.0 and np.mean(data) < 5.0)

        processed_bands = []
        for b in range(data.shape[0]):
            band = data[b]
            if not is_already_db and db_conversion:
                band = convert_to_db(band, epsilon=epsilon)
            if apply_filter:
                band = lee_speckle_filter(band, window_size=window_size)
            processed_bands.append(band)

        out_data = np.stack(processed_bands, axis=0).astype(np.float32)
        profile.update({
            'dtype': 'float32',
            'nodata': -9999.0
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(out_data)

    return {
        "source": input_path,
        "preprocessed_path": output_path,
        "db_conversion": db_conversion,
        "speckle_filter": "lee" if apply_filter else "none",
        "window_size": window_size,
        "bands_processed": data.shape[0],
    }
