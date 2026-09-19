import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class RSVQADataLoader:
    """
    DataLoader for RSVQA (Remote Sensing Visual Question Answering).
    Parses questions and answers across LR (Low Resolution) and HR (High Resolution) splits.
    """
    def __init__(self, annotation_path: str, images_dir: Optional[str] = None):
        self.annotation_path = Path(annotation_path)
        self.images_dir = Path(images_dir) if images_dir else self.annotation_path.parent
        self.samples: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        if not self.annotation_path.exists():
            return
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Accommodate standard RSVQA JSON schemas
        items = raw_data.get("questions", raw_data) if isinstance(raw_data, dict) else raw_data
        for item in items:
            q_text = item.get("question") or item.get("q")
            ans = str(item.get("answer") or item.get("a", ""))
            img_name = item.get("image_name") or item.get("image_id", "sample.tif")
            img_path = str((self.images_dir / img_name).resolve())
            q_type = item.get("type", "general")

            self.samples.append({
                "id": str(item.get("id", len(self.samples) + 1)),
                "question": q_text,
                "answer": ans,
                "image_path": img_path,
                "type": q_type,
                "prompt": f"User: <image>\nQuestion: {q_text}\nAssistant: ",
                "target": ans
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

class VRSBenchDataLoader:
    """
    DataLoader for VRSBench (Visual Referring Expression Benchmark for Remote Sensing).
    Handles text-guided region grounding with coordinates.
    """
    def __init__(self, annotation_path: str, images_dir: Optional[str] = None):
        self.annotation_path = Path(annotation_path)
        self.images_dir = Path(images_dir) if images_dir else self.annotation_path.parent
        self.samples: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        if not self.annotation_path.exists():
            return
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items = raw_data.get("annotations", raw_data) if isinstance(raw_data, dict) else raw_data
        for item in items:
            expr = item.get("expression") or item.get("query")
            bbox = item.get("bbox", [0.0, 0.0, 1.0, 1.0])  # [ymin, xmin, ymax, xmax]
            img_name = item.get("image_name") or item.get("image_id", "sample.tif")
            img_path = str((self.images_dir / img_name).resolve())

            # Format VLM target as normalized coordinate token sequence
            ymin, xmin, ymax, xmax = [round(float(c), 3) for c in bbox]
            target_str = f"<|box_start|>({ymin},{xmin}),({ymax},{xmax})<|box_end|>"

            self.samples.append({
                "id": str(item.get("id", len(self.samples) + 1)),
                "expression": expr,
                "bbox": [ymin, xmin, ymax, xmax],
                "image_path": img_path,
                "prompt": f"User: <image>\nLocate the region described by: {expr}\nAssistant: ",
                "target": target_str
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

class BigEarthNetDataLoader:
    """
    DataLoader for BigEarthNet multi-label remote sensing classification.
    """
    def __init__(self, metadata_path: str, images_dir: Optional[str] = None):
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir) if images_dir else self.metadata_path.parent
        self.samples: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        if not self.metadata_path.exists():
            return
        # If BigEarthNet.txt (line-separated patch_name and classes)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split(",")
                patch_name = parts[0].strip()
                labels = [p.strip() for p in parts[1:]] if len(parts) > 1 else ["Mixed rural landscape"]
                img_path = str((self.images_dir / f"{patch_name}.tif").resolve())
                target_str = ", ".join(labels)

                self.samples.append({
                    "id": f"ben_{idx + 1}",
                    "patch_name": patch_name,
                    "labels": labels,
                    "image_path": img_path,
                    "prompt": "User: <image>\nIdentify all land-cover and land-use categories present in this remote sensing patch.\nAssistant: ",
                    "target": target_str
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

class CDVQADataLoader:
    """
    DataLoader for CDVQA (Change Detection Visual Question Answering).
    Parses bi-temporal remote sensing image pairs (T1 and T2), change questions, and answers.
    """
    def __init__(self, annotation_path: str, images_dir: Optional[str] = None):
        self.annotation_path = Path(annotation_path)
        self.images_dir = Path(images_dir) if images_dir else self.annotation_path.parent
        self.samples: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        if not self.annotation_path.exists():
            return
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items = raw_data.get("questions", raw_data) if isinstance(raw_data, dict) else raw_data
        for item in items:
            q = item.get("question") or item.get("q")
            ans = str(item.get("answer") or item.get("a", ""))
            img_t1 = item.get("image_t1") or item.get("image_before", "delhi_optical.tif")
            img_t2 = item.get("image_t2") or item.get("image_after", "delhi_t2.tif")
            change_type = item.get("change_type", "general")
            has_change = item.get("has_change", True)

            t1_path = str((self.images_dir / img_t1).resolve()) if not Path(img_t1).is_absolute() else img_t1
            t2_path = str((self.images_dir / img_t2).resolve()) if not Path(img_t2).is_absolute() else img_t2

            self.samples.append({
                "id": str(item.get("id", len(self.samples) + 1)),
                "question": q,
                "answer": ans,
                "image_t1_path": t1_path,
                "image_t2_path": t2_path,
                "change_type": change_type,
                "has_change": has_change,
                "prompt": f"User: <image_t1><image_t2>\nBi-temporal Question: {q}\nAssistant: ",
                "target": ans
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
