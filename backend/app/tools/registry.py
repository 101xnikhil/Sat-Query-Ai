from typing import Dict, Any, List, Optional
from .base import BaseTool, ToolParameterError
from .vlm_vqa import RSVQATool
from .vlm_grounding import RSGroundingTool
from .vlm_caption import RSCaptionTool
from .change_vqa import ChangeVQATool
from .change_map import ChangeMapTool
from .fusion_segmenter import OpticalSARFusionTool
from .spectral_index import SpectralIndexTool

class ToolRegistry:
    """Central singleton registry mapping tool names to typed tool instances."""
    _instance: Optional["ToolRegistry"] = None

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_default_tools(self):
        tools = [
            RSVQATool(),
            RSCaptionTool(),
            RSGroundingTool(),
            ChangeVQATool(),
            ChangeMapTool(),
            OpticalSARFusionTool(),
            SpectralIndexTool(),
        ]
        for t in tools:
            self.register(t)

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered in ToolRegistry.")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "whitelisted_params": t.whitelisted_params,
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, params: Dict[str, Any]):
        tool = self.get(name)
        return tool.execute(params)
