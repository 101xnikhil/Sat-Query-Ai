import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import yaml

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class SiameseChangeNet(nn.Module):
        """
        Siamese Convolutional Network for pixel-level bi-temporal change detection.
        Extracts shared features from T1 and T2, computes difference representations,
        and predicts pixel-wise change probability map.
        """
        def __init__(self, in_channels: int = 4, feature_dim: int = 32):
            super().__init__()
            # Shared encoder branch
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )

            # Change decision head (takes |f1 - f2|, f1, f2)
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, 1, kernel_size=1)
            )

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            f1 = self.encoder(x1)
            f2 = self.encoder(x2)
            diff = torch.abs(f1 - f2)
            fused = torch.cat([diff, f1, f2], dim=1)
            logits = self.head(fused)
            return logits

class SiameseChangeDetector:
    """
    Inference wrapper for Siamese Change Detection.
    Handles neural model inference when weights are available, and robust
    Radiometric Change Vector Analysis (RCVA) in mock/CPU mode.
    """
    def __init__(self, config_path: Optional[str] = "configs/change_detection.yaml"):
        self.config = self._load_config(config_path)
        self.device = self.config.get("model", {}).get("device", "cpu")
        self.mock_mode = self.config.get("model", {}).get("mock_mode", True)
        self.threshold = self.config.get("inference", {}).get("threshold", 0.45)
        self.model = None

        if HAS_TORCH and not self.mock_mode:
            ckpt_path = self.config.get("model", {}).get("checkpoint_path")
            if ckpt_path and Path(ckpt_path).exists():
                try:
                    self.model = SiameseChangeNet(in_channels=4, feature_dim=32).to(self.device)
                    self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                    self.model.eval()
                except Exception:
                    self.model = None

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def predict_change_prob(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """
        Computes pixel-level change probability map [0.0, 1.0] from T1 and T2 arrays (C, H, W).
        """
        min_h = min(arr1.shape[1], arr2.shape[1])
        min_w = min(arr1.shape[2], arr2.shape[2])
        min_b = min(arr1.shape[0], arr2.shape[0])

        t1_crop = arr1[:min_b, :min_h, :min_w].astype(np.float32)
        t2_crop = arr2[:min_b, :min_h, :min_w].astype(np.float32)

        if self.model is not None and HAS_TORCH:
            with torch.no_grad():
                # Pad to 4 channels if necessary
                if min_b < 4:
                    pad = np.zeros((4 - min_b, min_h, min_w), dtype=np.float32)
                    t1_in = np.concatenate([t1_crop, pad], axis=0)
                    t2_in = np.concatenate([t2_crop, pad], axis=0)
                else:
                    t1_in, t2_in = t1_crop[:4], t2_crop[:4]

                t1_t = torch.from_numpy(t1_in).unsqueeze(0).to(self.device) / 255.0
                t2_t = torch.from_numpy(t2_in).unsqueeze(0).to(self.device) / 255.0

                logits = self.model(t1_t, t2_t)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                return prob

        # Fallback / Mock mode: Radiometric Change Vector Analysis (RCVA)
        diff = t2_crop - t1_crop
        mag = np.sqrt(np.sum(diff ** 2, axis=0))
        max_val = np.max(mag) if np.max(mag) > 0 else 1.0
        prob = (mag / max_val).astype(np.float32)
        return prob

    def detect_change(
        self,
        arr1: np.ndarray,
        arr2: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (binary_mask, probability_map).
        """
        t = threshold if threshold is not None else self.threshold
        prob = self.predict_change_prob(arr1, arr2)
        binary_mask = (prob > t).astype(np.uint8)
        return binary_mask, prob
