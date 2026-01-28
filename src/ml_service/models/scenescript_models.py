from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PointCloudData(BaseModel):
    points: List[List[float]]  # List of [x, y, z] coordinates
    nucleus_sampling_thresh: Optional[float] = 0.05
    verbose: Optional[bool] = True


class InferenceResponse(BaseModel):
    success: bool
    message: str
    entities: Optional[List[Dict[str, Any]]] = None
    processing_time: Optional[float] = None
    num_points: Optional[int] = None


class ModelStatus(BaseModel):
    loaded: bool
    model_path: Optional[str] = None
    device: Optional[str] = None
