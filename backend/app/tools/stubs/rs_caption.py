from typing import List, Type
from pydantic import BaseModel
from ..base import BaseTool
from ...schemas.tools import RSCaptionInput, RSCaptionOutput

class RSCaptionStubTool(BaseTool):
    name: str = "rs_caption"
    description: str = "Generates descriptive captions and domain tags for a remote sensing scene."
    input_schema: Type[BaseModel] = RSCaptionInput
    output_schema: Type[BaseModel] = RSCaptionOutput
    whitelisted_params: List[str] = ["image_path", "max_length", "style"]

    def run(self, input_data: RSCaptionInput) -> RSCaptionOutput:
        caption = (
            "A high-resolution remote sensing scene featuring mixed land-use: "
            "coastal water bodies, urban residential development, and contiguous agricultural zones."
        )
        tags = ["water_body", "urban_fabric", "agriculture", "coastal"]
        return RSCaptionOutput(
            caption=caption,
            tags=tags,
            confidence=0.91
        )
