import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

class BigEarthNetTxtDataLoader:
    """
    DataLoader for BigEarthNet.txt (arXiv:2603.29630).
    Multi-sensor remote sensing vision-language instruction following dataset.
    Supports official Parquet release, JSONL, and line-delimited metadata text formats.
    """
    HF_DATASET_URL = "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
    LICENSE = "Community Data License Agreement – Permissive – Version 1.0 (CDLA-Permissive-1.0)"

    def __init__(
        self,
        metadata_path: Optional[str] = None,
        annotation_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        split: Optional[str] = None
    ):
        self.split = split
        self.samples: List[Dict[str, Any]] = []
        effective_path = metadata_path or annotation_path

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                ben_cfg = cfg.get("datasets", {}).get("bigearthnet_txt", {})
                if not metadata_path:
                    metadata_path = ben_cfg.get("metadata_path")
                if not images_dir:
                    images_dir = ben_cfg.get("images_dir")

        self.metadata_path = Path(effective_path) if effective_path else None
        self.images_dir = Path(images_dir) if images_dir else (self.metadata_path.parent if self.metadata_path else None)

        if not self.metadata_path or not self.metadata_path.exists():
            self._print_manual_download_instructions()
            raise FileNotFoundError(
                f"BigEarthNet.txt annotation file not found at '{self.metadata_path}'. "
                f"Please follow the download instructions printed above."
            )

        self._load_data()

    def _print_manual_download_instructions(self):
        print("=" * 70, file=sys.stderr)
        print("📥 MANUAL DOWNLOAD REQUIRED FOR BigEarthNet.txt", file=sys.stderr)
        print(f"License: {self.LICENSE}", file=sys.stderr)
        print(f"Official Hugging Face Hub: {self.HF_DATASET_URL}", file=sys.stderr)
        print("Instructions:", file=sys.stderr)
        print("  1. Download via Hugging Face CLI or Python:", file=sys.stderr)
        print("     from datasets import load_dataset", file=sys.stderr)
        print("     ds = load_dataset('BIFOLD-BigEarthNetv2-0/BigEarthNet.txt')", file=sys.stderr)
        print("  2. Place 'BigEarthNet.txt.parquet' or text split in 'data/datasets/bigearthnet/'.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    def _load_data(self):
        ext = self.metadata_path.suffix.lower()

        # 1. Parquet format
        if ext == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(self.metadata_path)
                if self.split and "split" in df.columns:
                    df = df[df["split"] == self.split]
                for idx, row in df.iterrows():
                    patch_id = str(row.get("patch_id", row.get("ID", f"patch_{idx}")))
                    q_input = str(row.get("input", "Describe the land cover categories present."))
                    output_text = str(row.get("output", ""))
                    task_type = str(row.get("type", "vqa"))
                    img_path = str((self.images_dir / f"{patch_id}.tif").resolve()) if self.images_dir else patch_id

                    self.samples.append({
                        "id": str(row.get("ID", f"ben_{idx}")),
                        "patch_id": patch_id,
                        "instruction": q_input,
                        "target": output_text,
                        "type": task_type,
                        "image_path": img_path,
                        "prompt": f"User: <image>\n{q_input}\nAssistant: "
                    })
                return
            except ImportError:
                print("pandas/pyarrow not installed for reading parquet; falling back to line-by-line", file=sys.stderr)

        # 2. JSON / JSONL format
        if ext in [".json", ".jsonl"]:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                if ext == ".json":
                    data = json.load(f)
                    items = data.get("items", data) if isinstance(data, dict) else data
                else:
                    items = [json.loads(line) for line in f if line.strip()]

            for idx, item in enumerate(items):
                patch_id = item.get("patch_id") or item.get("patch_name", f"patch_{idx}")
                q_input = item.get("input") or "Identify land-use classes in this patch."
                output_text = item.get("output") or ", ".join(item.get("labels", ["Mixed landscape"]))
                img_path = str((self.images_dir / f"{patch_id}.tif").resolve()) if self.images_dir else patch_id

                self.samples.append({
                    "id": str(item.get("ID", f"ben_{idx + 1}")),
                    "patch_id": patch_id,
                    "instruction": q_input,
                    "target": output_text,
                    "type": item.get("type", "classification_vqa"),
                    "image_path": img_path,
                    "prompt": f"User: <image>\n{q_input}\nAssistant: "
                })
            return

        # 3. TSV / Text format (e.g. sample_bigearthnet.txt)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split(",")
                patch_name = parts[0].strip()
                labels = [p.strip() for p in parts[1:]] if len(parts) > 1 else ["Mixed rural landscape"]
                target_str = ", ".join(labels)
                img_path = str((self.images_dir / f"{patch_name}.tif").resolve()) if self.images_dir else patch_name

                inst = "Identify all land-cover and land-use categories present in this remote sensing patch."
                self.samples.append({
                    "id": f"ben_{idx + 1}",
                    "patch_id": patch_name,
                    "instruction": inst,
                    "input": inst,
                    "target": target_str,
                    "output": target_str,
                    "type": "captioning",
                    "image_path": img_path,
                    "prompt": f"User: <image>\n{inst}\nAssistant: "
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
