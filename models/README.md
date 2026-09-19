# Remote Sensing Model Adaptation & Fine-Tuning (SatQuery AI)

This directory provides model configs, inference engines, dataset loaders, and PEFT LoRA fine-tuning scripts for remote-sensing vision-language analysis (ISRO/SAC problem statement 26167).

---

## 1. Supported Base Models & Adaptation Architecture
- **Config file**: `models/configs/base_vlm.yaml`
- **Default Base Model**: `Qwen/Qwen2-VL-7B-Instruct` (or `google/paligemma-3b-pt-448` as a lightweight alternative).
- **Adaptation Strategy**: Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA):
  - Rank ($r$): `16`
  - Alpha ($\alpha$): `32`
  - Dropout: `0.05`
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

---

## 2. Ingestion & Benchmark Datasets
Loaders are located in `models/training/dataset_loaders.py`:
1. **RSVQA**: Remote Sensing Visual Question Answering (LR and HR splits).
2. **VRSBench**: Visual Referring Expression grounding & captioning for satellite imagery.
3. **BigEarthNet**: Sentinel-2 multispectral and Sentinel-1 SAR multi-label land-cover classifications.

Sample splits are provided in `data/datasets/` for immediate local verification and unit testing. Detailed download links and license notices are documented in `data/datasets/README.md`.

---

## 3. Running LoRA Fine-Tuning

### A. On External GPU (Google Colab / Kaggle / Cloud Instance)
Use the included standalone notebook: `models/training/colab_training_notebook.ipynb`, or run:
```bash
python models/training/train_lora.py \
    --config models/configs/base_vlm.yaml \
    --dataset rsvqa \
    --epochs 3 \
    --batch-size 2 \
    --output-dir models/checkpoints/vqa_lora
```

### B. Local Verification / CPU Run
```bash
python models/training/train_lora.py --dataset rsvqa --dry-run
python models/training/train_lora.py --dataset vrsbench --dry-run
python models/training/train_lora.py --dataset bigearthnet --dry-run
```

---

## 4. Quantitative Evaluation Benchmarks

Run real, un-fabricated evaluations on benchmark splits:

### Evaluate on RSVQA
```bash
python models/evaluation/eval_rsvqa.py \
    --annotations data/datasets/sample_rsvqa.json \
    --output data/outputs/eval_rsvqa_results.json
```
Computes:
- **Overall Accuracy**: Exact & semantic answer matching.
- **Category Accuracy**: `presence`, `comparison`, `count`, `general`.
- **Confidence Calibration**: Mean predicted token confidence.

### Evaluate on VRSBench
```bash
python models/evaluation/eval_vrsbench.py \
    --annotations data/datasets/sample_vrsbench.json \
    --output data/outputs/eval_vrsbench_results.json
```
Computes:
- **Mean Intersection-over-Union (mIoU)**.
- **Acc@0.50**: Precision at IoU $\ge 0.50$.
- **Acc@0.75**: Precision at IoU $\ge 0.75$.

---

## 5. Inference & Georeferencing
- `models/inference/vlm_wrapper.py`: Loads the model with optional LoRA adapter and calculates token logprob probabilities for verified confidence.
- `models/inference/georeferencer.py`: Projects detected bounding box coordinates onto the raster's affine transform matrix using `rasterio.transform.xy`, returning WGS84 (`EPSG:4326`) GeoJSON polygons.
