from typing import List
from pydantic import BaseModel, Field

class ValidatorResult(BaseModel):
    """
    Standard structured result returned by every validator.
    Contains ok status, warnings, errors, and actions_taken.
    """
    ok: bool = True
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)

    def merge(self, other: "ValidatorResult") -> "ValidatorResult":
        """Merge another ValidatorResult into this one."""
        if not other.ok:
            self.ok = False
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        self.actions_taken.extend(other.actions_taken)
        return self
