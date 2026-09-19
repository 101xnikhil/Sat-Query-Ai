import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.ndimage import label
import rasterio
from rasterio.transform import xy
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import pyproj

from backend.app.schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from backend.app.ingestion.reader import read_raster_array, read_image_metadata

def compute_radiometric_change(
    image_t1_path: str,
    image_t2_path: str,
    threshold: float = 0.45,
    min_area_m2: float = 100.0,
    output_dir: str = "data/outputs"
) -> Dict[str, Any]:
    """
    Performs Radiometric Change Vector Analysis (RCVA) between co-registered multi-temporal rasters.
    Computes exact empirical change area in km2, percentage changed, classified direction of change,
    exports georeferenced mask GeoTIFF, and returns vector GeoJSON features.
    """
    meta1 = read_image_metadata(image_t1_path)
    meta2 = read_image_metadata(image_t2_path)

    arr1 = read_raster_array(image_t1_path).astype(np.float32)
    arr2 = read_raster_array(image_t2_path).astype(np.float32)

    # Harmonize spatial dimensions
    min_h = min(arr1.shape[1], arr2.shape[1])
    min_w = min(arr1.shape[2], arr2.shape[2])
    min_bands = min(arr1.shape[0], arr2.shape[0])

    t1_slice = arr1[:min_bands, :min_h, :min_w]
    t2_slice = arr2[:min_bands, :min_h, :min_w]

    # 1. Multi-band Euclidean Change Vector Magnitude
    diff = t2_slice - t1_slice
    magnitude = np.sqrt(np.sum(diff ** 2, axis=0))

    # Normalize magnitude
    max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    norm_mag = magnitude / max_mag

    # Binary change mask from threshold
    raw_mask = (norm_mag > threshold).astype(np.uint8)

    # 2. Ground resolution in meters
    res_m = 10.0
    if meta1.resolution and len(meta1.resolution) >= 2:
        if meta1.is_georeferenced and meta1.crs and "4326" in meta1.crs:
            res_m = meta1.resolution[0] * 111320.0
        else:
            res_m = meta1.resolution[0]

    pixel_area_m2 = res_m * res_m

    # 3. Connected component filtering by min_area_m2
    labeled_array, num_features = label(raw_mask)
    clean_mask = np.zeros_like(raw_mask)

    for feat_id in range(1, num_features + 1):
        feat_pixels = np.sum(labeled_array == feat_id)
        feat_area_m2 = feat_pixels * pixel_area_m2
        if feat_area_m2 >= min_area_m2:
            clean_mask[labeled_array == feat_id] = 1

    total_pixels = float(min_h * min_w)
    changed_pixels = int(np.sum(clean_mask))
    percentage_changed = round((changed_pixels / total_pixels) * 100.0, 2)
    area_changed_m2 = changed_pixels * pixel_area_m2
    area_changed_km2 = round(area_changed_m2 / 1_000_000.0, 4)

    # 4. Classify Direction of Change using Spectral Indices
    direction = "no significant change"
    if changed_pixels > 0:
        mask_bool = clean_mask == 1
        eps = 1e-6

        if min_bands >= 4:
            # 0: Blue, 1: Green, 2: Red, 3: NIR
            nir1, red1, green1 = t1_slice[3], t1_slice[2], t1_slice[1]
            nir2, red2, green2 = t2_slice[3], t2_slice[2], t2_slice[1]

            ndvi1 = (nir1 - red1) / (nir1 + red1 + eps)
            ndvi2 = (nir2 - red2) / (nir2 + red2 + eps)
            delta_ndvi = float(np.mean(ndvi2[mask_bool] - ndvi1[mask_bool]))

            ndwi1 = (green1 - nir1) / (green1 + nir1 + eps)
            ndwi2 = (green2 - nir2) / (green2 + nir2 + eps)
            delta_ndwi = float(np.mean(ndwi2[mask_bool] - ndwi1[mask_bool]))

            if delta_ndvi < -0.15:
                direction = "vegetation loss / deforestation"
            elif delta_ndvi > 0.15:
                direction = "vegetation canopy growth"
            elif delta_ndwi < -0.15:
                direction = "water body recession / drying"
            elif delta_ndwi > 0.15:
                direction = "water level expansion / flooding"
            else:
                intensity_shift = np.mean(t2_slice[:, mask_bool]) - np.mean(t1_slice[:, mask_bool])
                direction = "new construction / surface brightness increase" if intensity_shift > 0 else "surface darkening / clearing"
        else:
            intensity_shift = np.mean(t2_slice[:, mask_bool]) - np.mean(t1_slice[:, mask_bool])
            direction = "construction / surface brightness increase" if intensity_shift > 0 else "vegetation decrease / darkening"

    # 5. Export Georeferenced GeoTIFF Change Mask
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_filename = f"change_mask_{meta1.image_id}_{meta2.image_id}.tif"
    mask_file_path = str(out_dir / mask_filename)

    with rasterio.open(image_t1_path) as src:
        profile = src.profile.copy()
        profile.update({
            'driver': 'GTiff',
            'height': min_h,
            'width': min_w,
            'count': 1,
            'dtype': rasterio.uint8,
            'nodata': 0
        })

    with rasterio.open(mask_file_path, 'w', **profile) as dst:
        dst.write(clean_mask, 1)

    # 6. Vectorize Changed Regions into GeoJSON Polygons
    geojson_features = []
    if changed_pixels > 0 and meta1.is_georeferenced:
        with rasterio.open(mask_file_path) as mask_src:
            geom_shapes = shapes(clean_mask, mask=clean_mask == 1, transform=mask_src.transform)
            transformer = None
            if mask_src.crs and mask_src.crs.to_string() != "EPSG:4326":
                try:
                    transformer = pyproj.Transformer.from_crs(mask_src.crs, "EPSG:4326", always_xy=True)
                except Exception:
                    transformer = None

            for idx, (geom, val) in enumerate(geom_shapes):
                if val == 1:
                    poly = shape(geom)
                    if poly.is_valid and poly.area > 0:
                        # Reproject coordinates to WGS84 if needed
                        poly_geojson = mapping(poly)
                        if transformer:
                            new_coords = []
                            for ring in poly_geojson["coordinates"]:
                                new_ring = [list(transformer.transform(x, y)) for x, y in ring]
                                new_coords.append(new_ring)
                            poly_geojson["coordinates"] = new_coords

                        feat = GeoJSONFeature(
                            type="Feature",
                            geometry=GeoJSONGeometry(type=poly_geojson["type"], coordinates=poly_geojson["coordinates"]),
                            properties={
                                "id": f"change_{idx + 1}",
                                "area_changed_km2": area_changed_km2,
                                "direction": direction
                            }
                        )
                        geojson_features.append(feat)
                        if len(geojson_features) >= 50:  # Cap at top 50 polygons for responsive rendering
                            break

    geojson = GeoJSONFeatureCollection(type="FeatureCollection", features=geojson_features) if geojson_features else None

    return {
        "change_mask_path": mask_file_path,
        "area_changed_km2": area_changed_km2,
        "area_changed_m2": round(area_changed_m2, 2),
        "percentage_changed": percentage_changed,
        "direction_of_change": direction,
        "changed_pixel_count": changed_pixels,
        "geojson": geojson
    }
