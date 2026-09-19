# Remote Sensing VLM LoRA Fine-Tuning Guide (Colab & Kaggle)

This folder contains the complete, standalone PEFT LoRA fine-tuning pipeline for adapting Vision-Language Models (e.g. Qwen2-VL, PaliGemma) on remote-sensing instruction datasets (RSVQA, VRSBench, BigEarthNet.txt).

---

## 1. Quickstart: Google Colab (Free T4 GPU)

1. Open a new notebook in [Google Colab](https://colab.research.google.com/) and enable GPU:
   `Runtime` -> `Change runtime type` -> **T4 GPU**
2. Clone repository & install requirements:
   ```bash
   !git clone https://github.com/101xnikhil/Sat-Query-Ai.git
   %cd Sat-Query-Ai
   !pip install -r models/train/requirements.txt
   ```
3. Run the LoRA training script:
   ```bash
   python models/train/finetune_vlm.py --config configs/train_lora.yaml
   ```
4. To run a quick smoke test on CPU or GPU:
   ```bash
   python models/train/finetune_vlm.py --config models/configs/tiny_cpu_vlm.yaml --smoke-test
   ```

---

## 2. Quickstart: Kaggle (2x T4 or P100)

1. Create a new Kaggle Notebook and enable GPU:
   `Settings` -> `Accelerator` -> **GPU T4 x2**
2. Install dependencies:
   ```bash
   !pip install -r models/train/requirements.txt
   ```
3. Execute fine-tuning:
   ```bash
   python models/train/finetune_vlm.py --config configs/train_lora.yaml
   ```

---

## 3. Configuration Parameters (`configs/train_lora.yaml`)

- `model.base_model`: Target Hugging Face VLM (`Qwen/Qwen2-VL-2B-Instruct` or `Qwen/Qwen2-VL-7B-Instruct`).
- `peft.r`: LoRA rank dimension (default: 16).
- `peft.lora_alpha`: LoRA alpha scaling (default: 32).
- `peft.target_modules`: Attention projection matrices (`["q_proj", "v_proj"]`).
- `datasets.mixture`: Configurable dataset paths and sampling weights:
  - RSVQA: 35%
  - VRSBench: 35%
  - BigEarthNet.txt: 30%

---

## 4. Resuming from Checkpoint

The script automatically detects previous checkpoints in `output_dir` or accepts `--resume_from_checkpoint`:
```bash
python models/train/finetune_vlm.py --config configs/train_lora.yaml --resume models/checkpoints/vqa_lora/checkpoint-step-100
```
