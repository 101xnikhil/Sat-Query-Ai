import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds

def create_demo_assets(target_dir: str = "data/demo_assets"):
    """
    Creates standalone synthetic and open-format demonstration assets for all 5 mandatory capabilities:
    1. Single-image VQA & Captioning (demo_single_optical.tif)
    2. Grounding (demo_single_optical.tif with distinct runway/oil tank signatures)
    3. Multitemporal Change Understanding (demo_temporal_t1.tif, demo_temporal_t2.tif)
    4. Optical + SAR Cross-Modal Analysis (demo_crossmodal_optical.tif, demo_crossmodal_sar.tif)
    5. Agentic Multi-Step Orchestration (all combined)
    """
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Common coordinate bounds: Bangalore region [77.55, 12.95, 77.60, 13.00]
    west, south, east, north = 77.55, 12.95, 77.60, 13.00
    w, h = 512, 512
    transform = from_bounds(west, south, east, north, w, h)
    crs = "EPSG:4326"

    print(f"Setting up SatQuery AI Demo Assets in '{out_dir}'...")

    # 1. Single Optical Scene (4 bands: B, G, R, NIR)
    # Features: Airfield runway strip, water reservoir, surrounding urban grid
    p_opt = out_dir / "demo_single_optical.tif"
    if not p_opt.exists():
        np.random.seed(42)
        b = np.random.randint(60, 100, size=(h, w), dtype=np.uint16)
        g = np.random.randint(80, 130, size=(h, w), dtype=np.uint16)
        r = np.random.randint(70, 120, size=(h, w), dtype=np.uint16)
        nir = np.random.randint(150, 220, size=(h, w), dtype=np.uint16)

        # Reservoir in southwest: high B/G, very low NIR
        b[300:460, 50:220] = 160
        g[300:460, 50:220] = 140
        r[300:460, 50:220] = 60
        nir[300:460, 50:220] = 20  # Low NIR = Water

        # Runway in northeast: high reflectance concrete
        b[80:120, 250:480] = 220
        g[80:120, 250:480] = 220
        r[80:120, 250:480] = 220
        nir[80:120, 250:480] = 210

        # Circular oil tanks
        for cx, cy in [(120, 150), (180, 150), (150, 190)]:
            y_i, x_i = np.ogrid[:h, :w]
            dist_sq = (x_i - cx)**2 + (y_i - cy)**2
            mask = dist_sq <= 18**2
            b[mask] = 230
            g[mask] = 230
            r[mask] = 230
            nir[mask] = 200

        with rasterio.open(
            str(p_opt), "w", driver="GTiff", height=h, width=w, count=4,
            dtype="uint16", crs=crs, transform=transform
        ) as dst:
            dst.write(b, 1)
            dst.write(g, 2)
            dst.write(r, 3)
            dst.write(nir, 4)
            dst.set_band_description(1, "Blue")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Red")
            dst.set_band_description(4, "NIR")
        print(f"  [+] Created {p_opt.name} (4 bands, 512x512, EPSG:4326)")

    # 2. Bi-temporal Pair (T1 and T2)
    # T1: 2021 pre-event baseline (rural/vegetation)
    # T2: 2023 post-event with 15% new urban construction cluster
    p_t1 = out_dir / "demo_temporal_t1.tif"
    p_t2 = out_dir / "demo_temporal_t2.tif"
    if not p_t1.exists() or not p_t2.exists():
        np.random.seed(101)
        base_t1 = np.random.randint(70, 110, size=(h, w), dtype=np.uint8)
        base_t2 = base_t1.copy()

        # Build significant urban development cluster in central zone (T2)
        base_t2[200:320, 200:360] = 245  # High reflectance built structures

        for p, arr, date_tag in [(p_t1, base_t1, "2021-02-15T10:00:00Z"), (p_t2, base_t2, "2023-04-10T10:00:00Z")]:
            with rasterio.open(
                str(p), "w", driver="GTiff", height=h, width=w, count=1,
                dtype="uint8", crs=crs, transform=transform
            ) as dst:
                dst.write(arr, 1)
                dst.update_tags(ACQUISITION_DATE=date_tag)
        print(f"  [+] Created {p_t1.name} and {p_t2.name} (Bi-temporal Pair with acquisition dates)")

    # 3. Optical + SAR Cross-Modal Pair
    p_opt_cm = out_dir / "demo_crossmodal_optical.tif"
    p_sar_cm = out_dir / "demo_crossmodal_sar.tif"
    if not p_opt_cm.exists() or not p_sar_cm.exists():
        np.random.seed(202)
        # Optical: cloudy over northern sector, clear lake in center
        opt_arr = np.zeros((4, h, w), dtype=np.uint16)
        opt_arr[:] = np.random.randint(80, 140, size=(4, h, w))
        # Surface lake: center [180:320, 180:340]
        opt_arr[0, 180:320, 180:340] = 180  # Blue
        opt_arr[1, 180:320, 180:340] = 160  # Green
        opt_arr[2, 180:320, 180:340] = 50   # Red
        opt_arr[3, 180:320, 180:340] = 15   # NIR (Water)
        # Built-up corridor: east [100:400, 380:480]
        opt_arr[0:3, 100:400, 380:480] = 210
        opt_arr[3, 100:400, 380:480] = 190

        # SAR: Backscatter in dB (Radar penetrates optical clouds)
        # Lake has low specular backscatter (-22 dB to -18 dB)
        # Built-up has cardinal double bounce (+2 dB to +8 dB)
        sar_arr = np.random.normal(loc=-12.0, scale=2.5, size=(1, h, w)).astype(np.float32)
        sar_arr[0, 180:320, 180:340] = np.random.normal(loc=-20.0, scale=1.0, size=(140, 160))  # Water specular
        sar_arr[0, 100:400, 380:480] = np.random.normal(loc=5.0, scale=1.8, size=(300, 100))    # Built-up double bounce

        with rasterio.open(
            str(p_opt_cm), "w", driver="GTiff", height=h, width=w, count=4,
            dtype="uint16", crs=crs, transform=transform
        ) as dst:
            dst.write(opt_arr)
            dst.update_tags(SENSOR="SENTINEL-2A", MODALITY="OPTICAL")

        with rasterio.open(
            str(p_sar_cm), "w", driver="GTiff", height=h, width=w, count=1,
            dtype="float32", crs=crs, transform=transform
        ) as dst:
            dst.write(sar_arr)
            dst.update_tags(SENSOR="SENTINEL-1B", MODALITY="SAR", POLARIZATION="VV")

        print(f"  [+] Created {p_opt_cm.name} and {p_sar_cm.name} (Optical-SAR Pair with dB backscatter)")

    print("✅ Demo Assets Setup Complete!")

if __name__ == "__main__":
    create_demo_assets()
