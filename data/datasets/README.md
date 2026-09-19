# Remote Sensing Datasets Documentation & Download Guide

This guide provides instructions for downloading and configuring the official remote-sensing datasets used by **SatQuery AI** (ISRO/SAC problem statement 26167).

---

## 1. CDVQA (Change Detection Visual Question Answering)
- **Description**: Benchmark for bi-temporal visual question answering over dual-epoch remote sensing imagery, pairing questions on urban expansion, vegetation dynamics, and water alterations with ground-truth change masks.
- **Paper / Source**: Yuan et al., *"A Benchmark Dataset for Change Detection-based Visual Question Answering in Remote Sensing"*.
- **Download Instructions**:
  1. Download official splits from the repository: https://github.com/ZhenghangYuan/CDVQA
  2. Extract image pairs ($T_1$ and $T_2$) and question-answer JSON files into `data/datasets/cdvqa/`:
     ```bash
     mkdir -p data/datasets/cdvqa
     # Place CDVQA_annotations.json and image pairs inside
     ```
  3. Point evaluation and training scripts to `data/datasets/cdvqa/`.

---

## 2. RSVQA (Remote Sensing Visual Question Answering)
- **Description**: The primary benchmark for VQA on Earth observation imagery, containing Low Resolution (Sentinel-2, 10m) and High Resolution (RGB aerial imagery, 15cm) splits.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Download Instructions**:
  1. Download official splits from Zenodo:
     - RSVQA LR (Low Resolution): https://zenodo.org/records/6344334
     - RSVQA HR (High Resolution): https://zenodo.org/records/6344367
  2. Extract the JSON annotations into `data/datasets/rsvqa/`.

---

## 3. VRSBench (Visual Referring Expression Grounding & Captioning)
- **Description**: Benchmark for text-guided region grounding, referring expression comprehension, and object captioning in high-resolution satellite imagery.
- **License**: Research-only Academic License / CC BY-NC-SA 4.0.
- **Download Instructions**:
  1. Access official repository: https://github.com/NJU-SIAT/VRSBench
  2. Download `VRSBench_annotations.json` and satellite image patches into `data/datasets/vrsbench/`.

---

## 4. BigEarthNet (Sentinel-2 & Sentinel-1 Benchmark)
- **Description**: Large-scale multi-modal benchmark containing 590,326 pairs of Sentinel-2 multispectral and Sentinel-1 dual-polarization SAR image patches.
- **License**: Community Data License Agreement – Permissive – Version 1.0 (CDLA-Permissive-1.0).
- **Download Instructions**:
  1. Register and download from official site: https://bigearth.net/
  2. Or download via HuggingFace: `datasets.load_dataset("bigearthnet")`
  3. Extract patches and train/val/test splits (`BigEarthNet.txt`) into `data/datasets/bigearthnet/`.

---

## 5. Immediate CI / Local Benchmark Splits
For quick local verification and running unit tests without multi-gigabyte downloads, the repository includes verified sample splits:
- `data/datasets/sample_cdvqa.json`
- `data/datasets/sample_rsvqa.json`
- `data/datasets/sample_vrsbench.json`
- `data/datasets/sample_bigearthnet.txt`
All evaluation and training scripts automatically fallback to these sample splits when full datasets are not present.
