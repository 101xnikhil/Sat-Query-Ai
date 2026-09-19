import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

class VRSBenchDataLoader:
    """
    DataLoader for VRSBench (Visual Referring Expression Benchmark for Remote Sensing).
    Supports captioning, VQA, and text-guided region grounding splits.
    """
    GITHUB_URL = "https://github.com/NJU-SIAT/VRSBench"
    LICENSE = "CC BY-NC-SA 4.0 (Non-Commercial, Share-Alike)"

    def __init__(
        self,
        task: str = "grounding",
        annotation_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        self.task = task.lower()
        self.samples: List[Dict[str, Any]] = []

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                vrs_cfg = cfg.get("datasets", {}).get("vrsbench", {})
                if not annotation_path:
                    annotation_path = vrs_cfg.get("annotation_path")
                if not images_dir:
                    images_dir = vrs_cfg.get("images_dir")

        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.images_dir = Path(images_dir) if images_dir else (self.annotation_path.parent if self.annotation_path else None)

        if not self.annotation_path or not self.annotation_path.exists():
            self._print_manual_download_instructions()
            raise FileNotFoundError(
                f"VRSBench annotation file not found at '{self.annotation_path}'. "
                f"Please follow the download instructions printed above."
            )

        self._load_data()

    def _print_manual_download_instructions(self):
        print("=" * 70, file=sys.stderr)
        print("📥 MANUAL DOWNLOAD REQUIRED FOR VRSBench", file=sys.stderr)
        print(f"License: {self.LICENSE}", file=sys.stderr)
        print(f"Official Repository: {self.GITHUB_URL}", file=sys.stderr)
        print("Instructions:", file=sys.stderr)
        print("  1. Clone or visit: https://github.com/NJU-SIAT/VRSBench", file=sys.stderr)
        print("  2. Request academic access if required by the license terms.", file=sys.stderr)
        print("  3. Download 'VRSBench_annotations.json' into 'data/datasets/vrsbench/'.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    def _load_data(self):
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = raw.get("annotations", raw) if isinstance(raw, dict) else raw
        for idx, item in enumerate(items):
            expr = item.get("expression") or item.get("query") or item.get("caption")
            bbox = item.get("bbox", [0.0, 0.0, 1.0, 1.0])  # [ymin, xmin, ymax, xmax]
            img_name = item.get("image_name") or item.get("image_id", "sample.tif")
            img_path = str((self.images_dir / img_name).resolve()) if self.images_dir else img_name
            answer = item.get("answer", "")

            ymin, xmin, ymax, xmax = [round(float(c), 3) for c in bbox]
            target_str = f"<|box_start|>({ymin},{xmin}),({ymax},{xmax})<|box_end|>"

            sample = {
                "id": str(item.get("id", f"vrsbench_{idx + 1}")),
                "task": self.task,
                "expression": expr,
                "bbox": [ymin, xmin, ymax, xmax],
                "answer": answer,
                "image_path": img_path,
                "prompt": f"User: <image>\nLocate: {expr}\nAssistant: ",
                "target": target_str if self.task == "grounding" else (answer or expr)
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
