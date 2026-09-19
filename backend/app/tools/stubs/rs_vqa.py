from typing import List, Type
from pydantic import BaseModel
from ..base import BaseTool
from ...schemas.tools import RSVQAInput, RSVQAOutput

class RSVQAStubTool(BaseTool):
    name: str = "rs_vqa"
    description: str = "Answers natural language questions regarding remote sensing imagery."
    input_schema: Type[BaseModel] = RSVQAInput
    output_schema: Type[BaseModel] = RSVQAOutput
    whitelisted_params: List[str] = ["query", "image_path", "confidence_threshold"]

    def run(self, input_data: RSVQAInput) -> RSVQAOutput:
        q = input_data.query.lower()
        if "water" in q or "river" in q or "lake" in q:
            ans = "The scene contains a prominent body of water with low backscatter and distinct shoreline boundaries."
            conf = 0.92
            evidence = "Surface water detected with characteristic low reflectance/backscatter signatures."
        elif "runway" in q or "airport" in q:
            ans = "An airport facility with two operational paved runways and taxiway systems is visible."
            conf = 0.89
            evidence = "Linear high-contrast geometric asphalt features matching aviation infrastructure."
        elif "urban" in q or "building" in q or "city" in q:
            ans = "Dense residential and commercial built-up structures occupy the central and eastern sectors."
            conf = 0.87
            evidence = "High spatial frequency rectilinear patterns and roof signatures."
        elif "vegetation" in q or "crop" in q or "forest" in q:
            ans = "Agricultural parcels and dense vegetative canopy predominate the southern sector."
            conf = 0.94
            evidence = "Strong near-infrared reflectance indicative of active photosynthetic biomass."
        else:
            ans = f"Analysis of the remote sensing image indicates terrain features directly addressing '{input_data.query}'."
            conf = 0.85
            evidence = "Extracted spectral and structural textural features across visible and infrared bands."

        return RSVQAOutput(
            answer=ans,
            confidence=conf,
            evidence_summary=evidence
        )
