from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
from pydantic import BaseModel, ValidationError as PydanticValidationError

class ToolParameterError(Exception):
    """Raised when an unwhitelisted or invalid parameter is passed to a tool."""
    pass

class BaseTool(ABC):
    """
    Abstract Base Class for SatQuery AI Tools.
    Enforces typed input/output validation and strict parameter whitelisting.
    """
    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    whitelisted_params: List[str]

    def validate_params(self, params: Dict[str, Any]) -> BaseModel:
        # Check for unwhitelisted parameters
        for key in params.keys():
            if key not in self.whitelisted_params:
                raise ToolParameterError(
                    f"Parameter '{key}' is not in the whitelist for tool '{self.name}'. "
                    f"Allowed parameters: {self.whitelisted_params}"
                )
        
        # Validate against pydantic input schema
        try:
            return self.input_schema(**params)
        except PydanticValidationError as e:
            raise ToolParameterError(f"Validation failed for tool '{self.name}': {str(e)}")

    @abstractmethod
    def run(self, validated_input: BaseModel) -> BaseModel:
        """Executes the tool logic and returns typed output."""
        pass

    def execute(self, params: Dict[str, Any]) -> BaseModel:
        """Validates parameters against whitelist and schema, then runs the tool."""
        validated_input = self.validate_params(params)
        return self.run(validated_input)
