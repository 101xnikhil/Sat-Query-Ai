# SatQuery AI: Limitations & Domain-Shift Notes

This document details the operational boundaries, physical sensor constraints, and domain-shift mitigation strategies for **SatQuery AI**, specifically evaluating transition from open baseline benchmarks (Sentinel-1/2, Landsat) to operational Indian Earth Observation sensors (**Cartosat-2S**, **RISAT-1A / EOS-04**).

---

## 1. Domain Shifts: Baseline Sensors vs. Indian EO Missions

| Dimension | Baseline Datasets (Sentinel-1/2, RSVQA, VRSBench) | Target Indian EO Assets (Cartosat-2S, RISAT-1A / EOS-04) | Physical & Algorithmic Impact | SatQuery Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Optical Ground Sampling Distance (GSD)** | 10m to 20m (Sentinel-2) / 30m (Landsat-8/9) | **0.65m Panchromatic**, **1.6m Multispectral (4 bands)** | Dramatic scale shift: individual vehicles, buildings, and tree crowns resolve as textured structures rather than mixed pixels. | Hierarchical tiling with $25\%$ overlap; affine bounding box projection scaled to pixel GSD; patch-based feature extraction. |
| **SAR Carrier & Polarization** | C-band (5.405 GHz), Dual-pol linear ($\text{VV} + \text{VH}$) (Sentinel-1) | **C-band (5.35 GHz)**, **Hybrid-Polarimetry (Circular Transmit, Dual Linear Receive - Compact Polarimetry)** | Linear VV/VH assumptions break when processing Compact Pol Stokes parameters ($S_0, S_1, S_2, S_3$); circular polarization exhibits distinct backscatter response to oriented urban geometries. | Modality validator accepts single-channel or multi-polarization GeoTIFFs; dynamic dB conversion $10\log_{10}(|x| + \epsilon)$; extensible Stokes parameter ingest hook. |
| **Radiometric Dynamic Range** | 12-bit / 16-bit integer, calibrated surface reflectance ($L2A$) | 10-bit / 11-bit (Cartosat-2S), 16-bit complex/amplitude (RISAT-1A) | Raw DN values require radiometric gain/bias normalization; sensor saturation over specular solar glint or metallic targets. | Robust min-max quantile stretch ($2\%$–$98\%$ clipping) before neural feature extraction; graceful degradation for uncalibrated DN rasters. |
| **Spectral Coverage** | 13 bands (including Coastal, Red-Edge, SWIR1, SWIR2) | 4 bands: Blue, Green, Red, Near-Infrared (NIR) | SWIR-dependent indices (MNDWI, NBR) cannot be computed directly from Cartosat-2S MX. | Automatic spectral fallback: defaults to Green/NIR NDWI (McFeeters) and Red/NIR NDVI when SWIR is unavailable. |

---

## 2. SAR Signal Phenomena & Preprocessing

### 2.1 Speckle Noise & Lee Adaptive Filtering
Synthetic Aperture Radar images are inherently degraded by granular speckle noise caused by coherent constructive and destructive interference of dephased backscatter waves within a resolution cell.

- **Effect on Models**: Raw high-frequency speckle causes spurious edge detections in change detection (RCVA) and misclassification in segmentation networks.
- **SatQuery Solution**: An automated **Lee Adaptive Filter** ($5\times 5$ moving window) is applied during the ingestion/validation phase:
  $$\hat{R} = \bar{I} + W \cdot (I - \bar{I})$$
  where weighting factor $W = \frac{\sigma_I^2 - \sigma_N^2}{\sigma_I^2}$, smoothing homogeneous clutter (water bodies, agricultural fields) while preserving high-gradient linear features (runways, coastlines, structural edges).

### 2.2 Geometric Distortions in Slant-to-Ground Range
- **Layover & Foreshortening**: In rugged or mountainous terrain (e.g., Himalayas, Western Ghats), steep slopes facing the radar sensor compress along-track returns or invert elevation ordering (layover).
- **Radar Shadow**: Regions obstructed by topography exhibit near-zero backscatter resembling specular water bodies.
- **Operational Constraint**: Without an external digital elevation model (CartoDEM), radar shadows can trigger false-positive water detections. SatQuery AI checks optical-SAR cross-tool agreement (NDWI spatial IoU) to detect this condition and applies confidence deductions when divergence is observed.

---

## 3. Optical Atmospheric Distortions & Monsoon Occlusion

### 3.1 Persistent Cloud Cover
During the Indian southwest and northeast monsoon seasons, cloud cover exceeds $80\%$ across the subcontinent for months at a time, completely blinding optical sensors like Cartosat-2S.

- **System Behavior**:
  1. Optical validators flag cloud-contaminated imagery based on high reflectance across visible channels and low spatial variance.
  2. The multi-modal planner automatically prioritizes SAR-based flood and water delineation (`optical_sar_fusion` running in SAR-dominant mode) when optical bands are compromised.
  3. The confidence scorer applies quality deductions (`quality_penalty = -0.15`) for cloud interference, transparently warning the analyst in the PDF briefing.

### 3.2 Haze and Aerosol Scattering
Sub-micron aerosols (Rayleigh and Mie scattering) in winter over the Indo-Gangetic Plains disproportionately scatter the Blue and Green bands, depressing contrast in optical imagery. SatQuery's quantile normalization and ratio-based spectral indices mitigate atmospheric attenuation across broad scenes.

---

## 4. Multi-Temporal Co-Registration & Alignment Constraints

Bi-temporal change analysis (`change_map`, `change_vqa`) and optical-SAR fusion require sub-pixel spatial congruence.

### 4.1 Grid Incompatibility Handling
1. **Extent Check**: The validator evaluates the spatial Intersection-over-Union ($\text{IoU}_{\text{spatial}}$) between bounding polygons in `EPSG:4326`:
   $$\text{IoU}_{\text{extent}} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$
   - If $\text{IoU}_{\text{extent}} < 0.70$, the query is aborted with a structured `IncompatibleInputError` preventing erroneous change calculations.
   - If $0.70 \le \text{IoU}_{\text{extent}} < 0.95$, a warning is logged in the trace and an automatic reprojection and resampling pass is triggered.
2. **Resampling Kernel Strategy**:
   - Continuous Optical bands: Bilinear or Bicubic convolution via rasterio `WarpedVRT`.
   - Discrete Segmentation Masks / SAR backscatter: Nearest-neighbor or average pooling to avoid introducing non-physical interpolated backscatter values.

---

## 5. Summary of Operational Limitations & Guidelines

| Condition | Observed Limitation | Recommended Operator Workflow |
| :--- | :--- | :--- |
| **Resolution Mismatch ($>5\times$)** | Combining 0.65m Cartosat-2S with 10m Sentinel-1 SAR causes boundary fuzziness around small structures. | Resample SAR to intermediate resolution (e.g. 2.5m) or rely on optical grounding for fine target boundaries. |
| **Off-Nadir Look Angles ($>25^\circ$)** | Tall urban structures in Cartosat-2S lean significantly, shifting apparent building footprints away from SAR double-bounce centroids. | Expand bounding box grounding tolerance; rely on multi-spectral indices (NDBI) alongside SAR. |
| **Specular Smooth Surfaces** | Dry asphalt runways and sand dunes exhibit specular radar reflection ($\le -18\text{ dB}$), mimicking calm water bodies in SAR. | Require optical verification (NDWI $\ge 0.1$) before confirming flood classification; review cross-tool agreement score. |
| **Target Size Below Sensor GSD** | Small vessels or vehicles ($< 5\text{m}$) cannot be grounded in medium-resolution Sentinel-2 imagery. | Grounding tool returns low-confidence warning when requested target dimensions fall below $3 \times \text{GSD}$. |
