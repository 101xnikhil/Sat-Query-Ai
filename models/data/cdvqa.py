import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

class CDVQADataLoader:
    """
    DataLoader for CDVQA (Change Detection Visual Question Answering).
    Constructed from the SECOND semantic change detection benchmark (2,968 bi-temporal pairs).
    Supports presence, direction ("increased"/"decreased"/"unchanged"), count, and comparison questions.
    """
    GITHUB_URL = "https://github.com/ZhenghangYuan/CDVQA"
    PAPER_URL = "https://arxiv.org/abs/2112.01314"
    LICENSE = "Academic Research / Non-Commercial Use (SECOND Benchmark)"

    def __init__(
        self,
        annotation_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        self.samples: List[Dict[str, Any]] = []

        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                cdvqa_cfg = cfg.get("datasets", {}).get("cdvqa", {})
                if not annotation_path:
                    annotation_path = cdvqa_cfg.get("annotation_path")
                if not images_dir:
                    images_dir = cdvqa_cfg.get("images_dir")
                if not masks_dir:
                    masks_dir = cdvqa_cfg.get("masks_dir")

        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.images_dir = Path(images_dir) if images_dir else (self.annotation_path.parent if self.annotation_path else None)
        self.masks_dir = Path(masks_dir) if masks_dir else (self.annotation_path.parent / "masks" if self.annotation_path else None)

        if not self.annotation_path or not self.annotation_path.exists():
            self._print_manual_download_instructions()
            raise FileNotFoundError(
                f"CDVQA annotation file not found at '{self.annotation_path}'. "
                f"Please follow the download instructions printed above."
            )

        self._load_data()

    def _print_manual_download_instructions(self):
        print("=" * 70, file=sys.stderr)
        print("📥 MANUAL DOWNLOAD REQUIRED FOR CDVQA (Change Detection VQA)", file=sys.stderr)
        print(f"License: {self.LICENSE}", file=sys.stderr)
        print(f"Official Repository: {self.GITHUB_URL}", file=sys.stderr)
        print(f"Publication: {self.PAPER_URL}", file=sys.stderr)
        print("Instructions:", file=sys.stderr)
        print("  1. Clone or visit: https://github.com/ZhenghangYuan/CDVQA", file=sys.stderr)
        print("  2. Download the SECOND change-detection image pairs (im1, im2) and semantic change masks.", file=sys.stderr)
        print("  3. Download CDVQA question-answer JSON annotations.", file=sys.stderr)
        print("  4. Place files in 'data/datasets/cdvqa/' and update configs/datasets.yaml.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    def _load_data(self):
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = raw.get("questions", raw) if isinstance(raw, dict) else raw
        for idx, item in enumerate(items):
            q_id = str(item.get("id", f"cdvqa_{idx + 1}"))
            question = item.get("question", "")
            answer = str(item.get("answer", "")).strip()
            change_type = item.get("change_type", item.get("type", "presence"))
            has_change = item.get("has_change", bool(answer.lower() not in ["no", "none", "unchanged", "0"]))

            im1_name = item.get("image_t1") or item.get("img1") or item.get("im1", "t1.tif")
            im2_name = item.get("image_t2") or item.get("img2") or item.get("im2", "t2.tif")
            mask_name = item.get("mask_path") or item.get("mask", f"mask_{q_id}.png")

            im1_path = str((self.images_dir / im1_name).resolve()) if self.images_dir else im1_name
            im2_path = str((self.images_dir / im2_name).resolve()) if self.images_dir else im2_name
            mask_path = str((self.masks_dir / mask_name).resolve()) if (self.masks_dir and (self.masks_dir / mask_name).exists()) else None

            prompt = (
                f"User: <image_t1><image_t2>\n"
                f"Analyze the bi-temporal change between Date T1 and Date T2.\n"
                f"Question: {question}\nAssistant: "
            )

            self.samples.append({
                "id": q_id,
                "question": question,
                "answer": answer,
                "change_type": change_type,
                "has_change": has_change,
                "image_t1_path": im1_path,
                "image_t2_path": im2_path,
                "mask_path": mask_path,
                "prompt": prompt,
                "target": answer
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
