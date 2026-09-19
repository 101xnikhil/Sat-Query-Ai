#!/usr/bin/env python3
"""
Generates synthetic sample remote sensing imagery and batch query test files.
"""
import os
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

def generate_samples(output_dir: str = "data/samples"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    width, height = 200, 200

    # Delhi coordinates: 77.0 to 77.1 E, 28.0 to 28.1 N
    minx, miny, maxx, maxy = 77.0, 28.0, 77.1, 28.1
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # 1. Optical 4-band (B, G, R, NIR)
    np.random.seed(42)
    b1 = np.random.randint(30, 80, (height, width), dtype=np.uint8)
    b2 = np.random.randint(40, 100, (height, width), dtype=np.uint8)
    b3 = np.random.randint(40, 120, (height, width), dtype=np.uint8)
    b4 = np.random.randint(120, 240, (height, width), dtype=np.uint8) # NIR
    opt_data = np.stack([b1, b2, b3, b4], axis=0)

    opt_path = out / "delhi_optical.tif"
    with rasterio.open(
        opt_path, 'w', driver='GTiff', height=height, width=width, count=4,
        dtype=rasterio.uint8, crs='EPSG:4326', transform=transform
    ) as dst:
        dst.write(opt_data)
    print(f"Generated: {opt_path}")

    # 2. SAR Amplitude 1-band (matching optical footprint)
    sar_data = np.random.gamma(shape=2.5, scale=12.0, size=(1, height, width)).astype(np.float32)
    # Simulate a low-backscatter water canal in bottom left
    sar_data[0, 120:180, 20:80] = np.random.uniform(0.1, 1.5, (60, 60))
    # Simulate high double-bounce built-up cluster in top right
    sar_data[0, 20:80, 120:180] = np.random.uniform(80.0, 220.0, (60, 60))

    sar_path = out / "delhi_sar.tif"
    with rasterio.open(
        sar_path, 'w', driver='GTiff', height=height, width=width, count=1,
        dtype=rasterio.float32, crs='EPSG:4326', transform=transform
    ) as dst:
        dst.write(sar_data)
    print(f"Generated: {sar_path}")

    # 3. Bi-Temporal T2 Optical (with simulated land use change)
    t2_data = opt_data.copy()
    # Urban development in top-right sector
    t2_data[0, 20:80, 120:180] = np.random.randint(140, 200, (60, 60))
    t2_data[1, 20:80, 120:180] = np.random.randint(140, 200, (60, 60))
    t2_data[2, 20:80, 120:180] = np.random.randint(150, 220, (60, 60))
    t2_data[3, 20:80, 120:180] = np.random.randint(60, 90, (60, 60)) # NIR dropped

    t2_path = out / "delhi_t2.tif"
    with rasterio.open(
        t2_path, 'w', driver='GTiff', height=height, width=width, count=4,
        dtype=rasterio.uint8, crs='EPSG:4326', transform=transform
    ) as dst:
        dst.write(t2_data)
    print(f"Generated: {t2_path}")

    # 4. Benchmark PNG (Non-georeferenced)
    rgb_arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    png_path = out / "benchmark_sample.png"
    Image.fromarray(rgb_arr).save(png_path)
    print(f"Generated: {png_path}")

    # 5. Sample Batch Queries JSON
    batch_queries = [
        {
            "query": "What is the primary land cover present in this scene?",
            "image_ids": [str(opt_path.resolve())]
        },
        {
            "query": "Where are the runways located?",
            "image_ids": [str(opt_path.resolve())]
        },
        {
            "query": "Describe the scene overview and structure",
            "image_ids": [str(opt_path.resolve())]
        },
        {
            "query": "Segment water and built-up areas using both optical and radar sensors",
            "image_ids": [str(opt_path.resolve()), str(sar_path.resolve())]
        },
        {
            "query": "What changed between date T1 and date T2?",
            "image_ids": [str(opt_path.resolve()), str(t2_path.resolve())]
        }
    ]
    batch_file = out / "sample_batch_queries.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_queries, f, indent=2)
    print(f"Generated: {batch_file}")

if __name__ == "__main__":
    generate_samples()
