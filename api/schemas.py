"""Pydantic request/response schemas."""
from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


class PredictionResponse(BaseModel):
    species: str
    probabilities: dict | None = None