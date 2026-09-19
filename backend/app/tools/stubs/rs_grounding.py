from typing import List, Type, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from ..base import BaseTool
from ...schemas.tools import RSGroundingInput, RSGroundingOutput
from ...schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from ...ingestion.reader import read_image_metadata

class RSGroundingStubTool(BaseTool):
    name: str = "rs_grounding"
    description: str = "Locates and grounds specified features/objects into GeoJSON bounding boxes or polygons."
    input_schema: Type[BaseModel] = RSGroundingInput
    output_schema: Type[BaseModel] = RSGroundingOutput
    whitelisted_params: List[str] = ["query", "image_path", "box_threshold", "text_threshold"]

    def run(self, input_data: RSGroundingInput) -> RSGroundingOutput:
        # Inspect image bounds to generate realistic coordinate polygons
        meta = read_image_metadata(input_data.image_path)
        features = []
        labels = []
        confidences = []

        if meta.bounds:
            minx, miny, maxx, maxy = meta.bounds.minx, meta.bounds.miny, meta.bounds.maxx, meta.bounds.maxy
            dx = (maxx - minx)
            dy = (maxy - miny)

            # Generate two grounded target regions based on query
            targets = [
                (minx + 0.2 * dx, miny + 0.2 * dy, minx + 0.45 * dx, miny + 0.45 * dy, 0.92),
                (minx + 0.55 * dx, miny + 0.55 * dy, minx + 0.8 * dx, miny + 0.8 * dy, 0.87),
            ]

            target_label = input_data.query.strip().split()[-1].capitalize()

            for i, (b_minx, b_miny, b_maxx, b_maxy, conf) in enumerate(targets):
                polygon_coords = [
                    [
                        [round(b_minx, 6), round(b_miny, 6)],
                        [round(b_maxx, 6), round(b_miny, 6)],
                        [round(b_maxx, 6), round(b_maxy, 6)],
                        [round(b_minx, 6), round(b_maxy, 6)],
                        [round(b_minx, 6), round(b_miny, 6)]
                    ]
                ]
                feature = GeoJSONFeature(
                    type="Feature",
                    geometry=GeoJSONGeometry(type="Polygon", coordinates=polygon_coords),
                    properties={
                        "id": f"grounding_{i+1}",
                        "label": f"{target_label} #{i+1}",
                        "confidence": conf,
                        "query": input_data.query
                    }
                )
                features.append(feature)
                labels.append(f"{target_label} #{i+1}")
                confidences.append(conf)

        return RSGroundingOutput(
            geojson=GeoJSONFeatureCollection(type="FeatureCollection", features=features),
            detected_count=len(features),
            labels=labels,
            confidences=confidences
        )
