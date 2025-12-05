# src/api/schemas.py
from pydantic import BaseModel, Field, root_validator, ValidationError
from typing import List, Optional, Dict, Any

class NamedFeatures(BaseModel):
    # dynamic: allow arbitrary keys (validated later)
    __root__: Dict[str, float]

class ArrayFeatures(BaseModel):
    features: List[float] = Field(..., description="Ordered list of numeric features")

class PredictionResponse(BaseModel):
    predicted_label: int
    predicted_proba: List[float]
    model_name: Optional[str] = "breast-cancer-rf"
