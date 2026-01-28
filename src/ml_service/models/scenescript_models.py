from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PointCloudData(BaseModel):
    # Use Field for defaults to be explicit
    points: List[List[float]]
    nucleus_sampling_thresh: Optional[float] = Field(default=0.05)
    verbose: Optional[bool] = Field(default=True)


class InferenceResponse(BaseModel):
    success: bool
    message: str
    entities: Optional[List[Dict[str, Any]]] = Field(default=None)
    processing_time: Optional[float] = Field(default=None)
    num_points: Optional[int] = Field(default=None)


class ModelStatus(BaseModel):
    loaded: bool
    model_path: Optional[str] = Field(default=None)
    device: Optional[str] = Field(default=None)
