import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

class RSVQADataLoader:
    """
    DataLoader for RSVQA (Remote Sensing Visual Question Answering).
    Supports LR (Low Resolution, Sentinel-2) and HR (High Resolution, Aerial RGB) splits.
    Reads configuration from YAML or direct parameters.
    """
    ZENODO_LR_URL = "https://zenodo.org/records/6344334"
    ZENODO_HR_URL = "https://zenodo.org/records/6344367"
    LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

    def __init__(
        self,
        split: str = "LR",
        annotation_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        self.split = split.upper()
        self.samples: List[Dict[str, Any]] = []
        
        # Load from config if available
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                rsvqa_cfg = cfg.get("datasets", {}).get("rsvqa", {})
                sub_cfg = rsvqa_cfg.get("lr_split" if self.split == "LR" else "hr_split", {})
                if not annotation_path:
                    annotation_path = sub_cfg.get("annotation_path")
                if not images_dir:
                    images_dir = sub_cfg.get("images_dir")

        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.images_dir = Path(images_dir) if images_dir else (self.annotation_path.parent if self.annotation_path else None)

        if not self.annotation_path or not self.annotation_path.exists():
            self._print_manual_download_instructions()
            raise FileNotFoundError(
                f"RSVQA ({self.split}) annotation file not found at '{self.annotation_path}'. "
                f"Please follow the download instructions printed above."
            )

        self._load_data()

    def _print_manual_download_instructions(self):
        url = self.ZENODO_LR_URL if self.split == "LR" else self.ZENODO_HR_URL
        print("=" * 70, file=sys.stderr)
        print(f"📥 MANUAL DOWNLOAD REQUIRED FOR RSVQA ({self.split})", file=sys.stderr)
        print(f"License: {self.LICENSE}", file=sys.stderr)
        print(f"Official Download URL: {url}", file=sys.stderr)
        print(f"Instructions:", file=sys.stderr)
        print(f"  1. Visit the Zenodo repository: {url}", file=sys.stderr)
        print(f"  2. Download 'Questions_{self.split}.json' and 'Answers_{self.split}.json' (or all_questions.json).", file=sys.stderr)
        print(f"  3. Place them inside 'data/datasets/rsvqa/' or update configs/datasets.yaml.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    def _load_data(self):
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = raw.get("questions", raw) if isinstance(raw, dict) else raw
        for idx, item in enumerate(items):
            q_text = item.get("question") or item.get("q")
            ans = str(item.get("answer") or item.get("a", ""))
            img_name = item.get("image_name") or item.get("image_id", "sample.tif")
            img_path = str((self.images_dir / img_name).resolve()) if self.images_dir else img_name
            q_type = item.get("type", "general")

            self.samples.append({
                "id": str(item.get("id", f"rsvqa_{self.split}_{idx + 1}")),
                "split": self.split,
                "question": q_text,
                "answer": ans,
                "image_path": img_path,
                "type": q_type,
                "prompt": f"User: <image>\nQuestion: {q_text}\nAssistant: ",
                "target": ans
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
