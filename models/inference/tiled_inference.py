import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np

from .georeferencer import BoundingBox2D

class TileWindow:
    def __init__(self, col_off: int, row_off: int, width: int, height: int, tile_id: int):
        self.col_off = col_off
        self.row_off = row_off
        self.width = width
        self.height = height
        self.tile_id = tile_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "col_off": self.col_off,
            "row_off": self.row_off,
            "width": self.width,
            "height": self.height
        }

def generate_tiles(
    img_width: int,
    img_height: int,
    tile_size: int = 512,
    overlap: int = 64
) -> List[TileWindow]:
    """
    Generates regular overlapping tile windows covering the full raster extent.
    If image dimensions are smaller than tile_size, returns a single tile.
    """
    if img_width <= tile_size and img_height <= tile_size:
        return [TileWindow(0, 0, img_width, img_height, tile_id=1)]

    stride = max(32, tile_size - overlap)
    tiles: List[TileWindow] = []
    tile_count = 0

    y = 0
    while y < img_height:
        h = min(tile_size, img_height - y)
        x = 0
        while x < img_width:
            w = min(tile_size, img_width - x)
            tile_count += 1
            tiles.append(TileWindow(col_off=x, row_off=y, width=w, height=h, tile_id=tile_count))
            if x + w >= img_width:
                break
            x += stride
        if y + h >= img_height:
            break
        y += stride

    return tiles

def compute_box_iou(b1: Any, b2: Any) -> float:
    """Computes standard 2D Intersection-over-Union (IoU) between BoundingBox2D or [ymin, xmin, ymax, xmax] lists."""
    def _extract_coords(b):
        if hasattr(b, "ymin"):
            return b.ymin, b.xmin, b.ymax, b.xmax
        return b[0], b[1], b[2], b[3]

    y1_min, x1_min, y1_max, x1_max = _extract_coords(b1)
    y2_min, x2_min, y2_max, x2_max = _extract_coords(b2)

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    if inter_ymax <= inter_ymin or inter_xmax <= inter_xmin:
        return 0.0

    inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
    area1 = (y1_max - y1_min) * (x1_max - x1_min)
    area2 = (y2_max - y2_min) * (x2_max - x2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)

def nms_boxes(
    boxes: List[BoundingBox2D],
    iou_threshold: float = 0.50
) -> List[BoundingBox2D]:
    """
    Greedy Non-Maximum Suppression (NMS) merging duplicate candidate detections.
    Sorts boxes by confidence descending and removes boxes with IoU >= iou_threshold.
    """
    if not boxes:
        return []

    # Sort descending by confidence
    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept_boxes: List[BoundingBox2D] = []

    for candidate in sorted_boxes:
        suppress = False
        for kept in kept_boxes:
            if compute_box_iou(candidate, kept) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept_boxes.append(candidate)

    return kept_boxes

def aggregate_vqa_tile_answers(
    tile_results: List[Dict[str, Any]],
    query: str
) -> Dict[str, Any]:
    """
    Documented VQA Tile Aggregation Strategy:
    1. Count Queries ('how many', 'count', 'number of'):
       Sums numeric counts extracted from detection counts across tiles.
    2. Binary Yes/No Queries ('is there', 'are there', 'does', 'has', 'present'):
       Majority vote across tiles. Confidence is the mean confidence of the majority votes.
    3. Open-Ended Descriptive Queries:
       Selects answer from the tile with the highest token-probability confidence.
    """
    if not tile_results:
        return {
            "answer": "No features detected across tiles.",
            "confidence": 0.50,
            "evidence": "Zero tiles processed.",
            "tiles_used": []
        }

    q = query.lower()
    tiles_used = [t.get("tile_info", {}) for t in tile_results]

    # 1. Count Queries
    is_count_query = any(re.search(rf"\b{k}\b", q) for k in ["how many", "count", "number of"])
    if is_count_query:
        total_count = 0
        confidences = []
        for t in tile_results:
            # Check for detected_count or integer inside answer
            cnt = t.get("detected_count", 0)
            if cnt == 0:
                nums = re.findall(r"\b\d+\b", t.get("answer", ""))
                if nums:
                    cnt = int(nums[0])
            total_count += cnt
            confidences.append(t.get("confidence", 0.85))

        mean_conf = round(float(np.mean(confidences)), 4) if confidences else 0.85
        return {
            "answer": f"There are {total_count} detected instances in the scene.",
            "confidence": mean_conf,
            "evidence": f"Aggregated count summation across {len(tile_results)} tiles.",
            "tiles_used": tiles_used,
            "detected_count": total_count
        }

    # 2. Binary Yes/No Queries
    is_binary_query = any(re.search(rf"\b{k}\b", q) for k in ["is there", "are there", "does", "has", "visible", "present"])
    if is_binary_query:
        votes = []
        conf_map = {"yes": [], "no": []}
        for t in tile_results:
            ans_lower = t.get("answer", "").lower()
            if "yes" in ans_lower or "present" in ans_lower or "detected" in ans_lower:
                votes.append("yes")
                conf_map["yes"].append(t.get("confidence", 0.85))
            else:
                votes.append("no")
                conf_map["no"].append(t.get("confidence", 0.85))

        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]
        winning_confs = conf_map[winner]
        mean_conf = round(float(np.mean(winning_confs)), 4) if winning_confs else 0.85

        answer_text = (
            f"Yes, the target feature is confirmed across {vote_counts['yes']} tile sectors."
            if winner == "yes" else
            "No, the target feature was not identified in the analyzed scene."
        )
        return {
            "answer": answer_text,
            "confidence": mean_conf,
            "evidence": f"Majority voting: {vote_counts['yes']} Yes vs {vote_counts['no']} No across {len(tile_results)} tiles.",
            "tiles_used": tiles_used
        }

    # 3. Open-Ended Descriptive Queries -> Top Confidence
    sorted_by_conf = sorted(tile_results, key=lambda t: t.get("confidence", 0.0), reverse=True)
    top_tile = sorted_by_conf[0]

    return {
        "answer": top_tile.get("answer", ""),
        "confidence": top_tile.get("confidence", 0.85),
        "evidence": f"Selected highest-confidence tile answer ({top_tile.get('confidence')}) among {len(tile_results)} tiles. {top_tile.get('evidence', '')}",
        "tiles_used": tiles_used
    }
