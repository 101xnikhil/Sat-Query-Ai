from typing import List, Type
from pydantic import BaseModel
from ..base import BaseTool
from ...schemas.tools import ChangeVQAInput, ChangeVQAOutput

class ChangeVQAStubTool(BaseTool):
    name: str = "change_vqa"
    description: str = "Answers bi-temporal comparative questions between two acquisition dates."
    input_schema: Type[BaseModel] = ChangeVQAInput
    output_schema: Type[BaseModel] = ChangeVQAOutput
    whitelisted_params: List[str] = ["query", "image_t1_path", "image_t2_path"]

    def run(self, input_data: ChangeVQAInput) -> ChangeVQAOutput:
        q = input_data.query.lower()
        if "water" in q or "reservoir" in q:
            ans = "A 14.8% decrease in water reservoir surface area is observed between date T1 and date T2."
            summary = "Significant shoreline retreat and drying of peripheral embayments."
            conf = 0.91
        elif "urban" in q or "construction" in q or "building" in q:
            ans = "New commercial and infrastructure development expanded across 3.42 sq km along the western corridor."
            summary = "Conversion of previously unpaved bare soil into structured impermeable surfaces."
            conf = 0.88
        elif "forest" in q or "vegetation" in q or "deforestation" in q:
            ans = "Vegetation index decline detected across 5.10 sq km indicating seasonal harvesting or clearing."
            summary = "Marked spectral shift from dense vegetative canopy to exposed soil."
            conf = 0.93
        else:
            ans = f"Bi-temporal comparative analysis for query '{input_data.query}' identifies localized morphological alterations."
            summary = "Pixel difference analysis reveals multi-modal spectral variance between temporal epochs."
            conf = 0.86

        return ChangeVQAOutput(
            answer=ans,
            confidence=conf,
            change_summary=summary
        )
