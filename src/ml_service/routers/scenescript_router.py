import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ml_service.dependencies import require_bearer
from ml_service.models.scenescript_models import InferenceResponse, ModelStatus, PointCloudData
from ml_service.startup import SceneScriptState

router = APIRouter(
    prefix="/scenescript", tags=["scenescript"], dependencies=[Depends(require_bearer)]
)


def serialize_entity_params(params):
    """Convert entity parameters to JSON-serializable format"""
    serialized = {}
    for key, value in params.items():
        if torch.is_tensor(value):
            serialized[key] = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized


def _get_pointcloud_class(base_dir: str):
    """
    Lazily import the SceneScript PointCloud class using the configured base_dir.
    """
    scenescript_root = base_dir  # .../reinvent-ml-server/scenescript
    if scenescript_root not in sys.path:
        sys.path.insert(0, scenescript_root)
    try:
        # Scenescript code expects `src` as a top-level package
        from src.data.point_cloud import PointCloud  # type: ignore[import]
    except ImportError as e:  # pragma: no cover - optional dependency
        raise HTTPException(
            status_code=503,
            detail=f"SceneScript PointCloud not available: {e}",
        )
    return PointCloud


def get_scenescript_state(request: Request) -> SceneScriptState:
    """
    Retrieve SceneScript state from app.state.

    Raises 503 if the model is not available so that endpoints can
    rely on a fully initialized state.
    """
    state: SceneScriptState | None = getattr(request.app.state, "scenescript", None)
    if state is None or state.model is None:
        raise HTTPException(
            status_code=503,
            detail="SceneScript model not loaded. Please check model status.",
        )
    return state


@router.get("/")
def read_root():
    return {"message": "SceneScript API is running", "status": "healthy"}


@router.get("/model/status", response_model=ModelStatus)
def get_model_status(request: Request) -> ModelStatus:
    """Get the current model loading status."""
    state: SceneScriptState | None = getattr(request.app.state, "scenescript", None)
    loaded = bool(state and state.model is not None)
    device = getattr(state, "device", None) if state else None
    model_path = getattr(state, "weights_path", None) if state else None
    if loaded and device is None:
        # Fallback to runtime check if device was not recorded
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ModelStatus(loaded=loaded, model_path=model_path, device=device)


@router.post("/inference", response_model=InferenceResponse)
def run_inference(
    data: PointCloudData,
    state: SceneScriptState = Depends(get_scenescript_state),
):
    """Run SceneScript inference on point cloud data"""
    try:
        import time

        start_time = time.time()

        # Convert input points to numpy array
        points = np.array(data.points, dtype=np.float32)

        if points.shape[1] != 3:
            raise HTTPException(status_code=400, detail="Points must have 3 coordinates [x, y, z]")

        # Run inference
        lang_seq = state.model.run_inference(  # type: ignore[call-arg]
            points,
            nucleus_sampling_thresh=data.nucleus_sampling_thresh,
            verbose=data.verbose,
        )

        processing_time = time.time() - start_time

        # Convert entities to serializable format
        entities = []
        if lang_seq and hasattr(lang_seq, "entities"):
            for entity in lang_seq.entities:
                entity_dict = {
                    "command": entity.COMMAND_STRING,
                    "params": serialize_entity_params(entity.params),
                    "id": int(entity.params.get("id", -1)),
                }
                entities.append(entity_dict)

        return InferenceResponse(
            success=True,
            message="Inference completed successfully",
            entities=entities,
            processing_time=processing_time,
            num_points=len(points),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/inference/file", response_model=InferenceResponse)
def run_inference_from_uploaded_file(
    filename: str = Form(...),
    nucleus_sampling_thresh: float = Form(0.05),
    verbose: bool = Form(True),
    state: SceneScriptState = Depends(get_scenescript_state),
):
    """Run SceneScript inference on previously uploaded point cloud file"""
    try:
        import time

        start_time = time.time()

        # Construct the file path in the pointclouds directory
        file_path = os.path.join(state.pointcloud_dir, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Point cloud file not found: {filename}")

        # Load point cloud using the SceneScript PointCloud class
        PointCloud = _get_pointcloud_class(state.base_dir)
        point_cloud_obj = PointCloud.load_from_file(file_path)
        points = point_cloud_obj.points

        # Convert numpy array to torch tensor as expected by run_inference
        import torch

        if not torch.is_tensor(points):
            points = torch.FloatTensor(points)

        # Move to device if CUDA is available
        if torch.cuda.is_available() and hasattr(state.model, "device"):
            points = points.to(state.model.device)  # type: ignore[attr-defined]

        # Run inference using the model's run_inference method
        lang_seq = state.model.run_inference(  # type: ignore[call-arg]
            raw_point_cloud=points,
            nucleus_sampling_thresh=nucleus_sampling_thresh,
            verbose=verbose,
        )

        processing_time = time.time() - start_time

        # Convert entities to serializable format
        entities = []
        if lang_seq and hasattr(lang_seq, "entities"):
            for entity in lang_seq.entities:
                entity_dict = {
                    "command": entity.COMMAND_STRING,
                    "params": serialize_entity_params(entity.params),
                    "id": int(entity.params.get("id", -1)),
                }
                entities.append(entity_dict)

        return InferenceResponse(
            success=True,
            message="Inference completed successfully",
            entities=entities,
            processing_time=processing_time,
            num_points=len(points),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/inference/scene", response_model=InferenceResponse)
def run_inference_on_scene_file(
    file: UploadFile = File(...),  # File from FormData
    nucleus_sampling_thresh: float = Form(0.05),  # FormData field
    verbose: bool = Form(True),  # FormData field
    state: SceneScriptState = Depends(get_scenescript_state),
):
    """Run SceneScript inference on uploaded scene file"""
    try:
        import time

        start_time = time.time()
        # Save uploaded file temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and process the point cloud
        PointCloud = _get_pointcloud_class(state.base_dir)
        point_cloud_obj = PointCloud.load_from_file(temp_file_path)
        raw_point_cloud = torch.tensor(point_cloud_obj.points, dtype=torch.float32)

        # Run inference
        lang_seq = state.model.run_inference(  # type: ignore[call-arg]
            raw_point_cloud=raw_point_cloud,
            nucleus_sampling_thresh=nucleus_sampling_thresh,
            verbose=verbose,
        )

        # Clean up temp file
        os.remove(temp_file_path)

        processing_time = time.time() - start_time

        # Convert entities to serializable format
        entities = []
        if lang_seq and hasattr(lang_seq, "entities"):
            for entity in lang_seq.entities:
                entity_dict = {
                    "command": entity.COMMAND_STRING,
                    "params": serialize_entity_params(entity.params),
                    "id": int(entity.params.get("id", -1)),
                }
                entities.append(entity_dict)

        return InferenceResponse(
            success=True,
            message="Inference completed successfully",
            entities=entities,
            processing_time=processing_time,
            num_points=len(point_cloud_obj.points),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/pointclouds")
def list_available_pointclouds(request: Request):
    """List available point cloud files."""
    state: SceneScriptState | None = getattr(request.app.state, "scenescript", None)
    if not state:
        raise HTTPException(status_code=503, detail="SceneScript state not initialized.")

    pointclouds_dir = state.pointcloud_dir
    try:
        files = [f for f in os.listdir(pointclouds_dir) if f.endswith((".csv", ".csv.gz"))]
        return {"available_files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}") from e


@router.post("/upload")
async def upload_files(
    request: Request,
    semidense_points: UploadFile = File(None),
    trajectory: UploadFile = File(None),
):
    """Upload and store point cloud and trajectory files."""
    state: SceneScriptState | None = getattr(request.app.state, "scenescript", None)
    if not state:
        raise HTTPException(status_code=503, detail="SceneScript state not initialized.")

    uploads_dir = state.uploads_dir
    pointclouds_dir = state.pointcloud_dir
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(uploads_dir, exist_ok=True)

        uploaded_files = {}

        # Handle semidense points file
        if semidense_points:
            points_filename = (
                f"uploaded_semidense_points_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv.gz"
            )
            points_path = os.path.join(uploads_dir, points_filename)

            with open(points_path, "wb") as buffer:
                content = await semidense_points.read()
                buffer.write(content)

            uploaded_files["semidense_points"] = {
                "filename": points_filename,
                "path": points_path,
                "size": len(content),
                "original_name": semidense_points.filename,
            }

        # Handle trajectory file
        if trajectory:
            traj_filename = (
                f"uploaded_trajectory_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            traj_path = os.path.join(uploads_dir, traj_filename)

            with open(traj_path, "wb") as buffer:
                content = await trajectory.read()
                buffer.write(content)

            uploaded_files["trajectory"] = {
                "filename": traj_filename,
                "path": traj_path,
                "size": len(content),
                "original_name": trajectory.filename,
            }

        # Also copy semidense points to pointclouds directory for inference
        if semidense_points:
            os.makedirs(pointclouds_dir, exist_ok=True)

            inference_filename = f"uploaded_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv.gz"
            inference_path = os.path.join(pointclouds_dir, inference_filename)

            # Copy the file for inference
            with (
                open(inference_path, "wb") as buffer,
                open(uploaded_files["semidense_points"]["path"], "rb") as source,
            ):
                buffer.write(source.read())

            uploaded_files["inference_file"] = {
                "filename": inference_filename,
                "path": inference_path,
            }

        return {
            "success": True,
            "message": "Files uploaded successfully",
            "uploaded_files": uploaded_files,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e


@router.get("/uploads")
def list_uploaded_files(request: Request):
    """List all uploaded files."""
    state: SceneScriptState | None = getattr(request.app.state, "scenescript", None)
    if not state:
        raise HTTPException(status_code=503, detail="SceneScript state not initialized.")

    uploads_dir = state.uploads_dir
    try:
        if not os.path.exists(uploads_dir):
            return {"uploaded_files": []}

        files = []
        for f in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, f)
            if os.path.isfile(file_path):
                files.append(
                    {
                        "filename": f,
                        "size": os.path.getsize(file_path),
                        "modified": pd.Timestamp.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat(),
                    }
                )

        return {"uploaded_files": files}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing uploaded files: {str(e)}"
        ) from e


@router.delete("/uploads/{filename}")
def delete_uploaded_file(filename: str, request: Request):
    """Delete an uploaded file."""
    state: SceneScriptState | None = getattr(request.app.state, "scenescript", None)
    if not state:
        raise HTTPException(status_code=503, detail="SceneScript state not initialized.")

    uploads_dir = state.uploads_dir
    file_path = os.path.join(uploads_dir, filename)

    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)
        return {"success": True, "message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}") from e
