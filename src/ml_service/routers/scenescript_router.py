import os
import shutil

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from models.scenescript_models import InferenceResponse, ModelStatus, PointCloudData

# Try to import SceneScript modules with error handling
try:
    # from ....scenescript.src.data.language_sequence import LanguageSequence
    from ....scenescript.src.data.point_cloud import PointCloud
    from ....scenescript.src.networks.scenescript_model import SceneScriptWrapper

    SCENESCRIPT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SceneScript modules not available: {e}")
    SCENESCRIPT_AVAILABLE = False

router = APIRouter(path="/scenescript")

# Global model instance
model_wrapper = None  # Add to lifespan or something
SCENESCRIPT_DIR = ""
UPLOADS_DIR = ""
STORAGE_DIR = ""


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


@router.on_event("startup")
async def load_model():
    """Load the SceneScript model on startup"""
    global model_wrapper

    if not SCENESCRIPT_AVAILABLE:
        print("SceneScript modules not available. Model loading skipped.")
        return

    try:
        # Change to the scenescript directory
        os.chdir("/home/playboigeorgy/scenescript_clone_v3")

        ckpt_path = "./weights/scenescript_model_ase.ckpt"
        if not os.path.exists(ckpt_path):
            print(f"Warning: Model checkpoint not found at {ckpt_path}")
            return

        print("Loading SceneScript model...")
        if torch.cuda.is_available():
            model_wrapper = SceneScriptWrapper.load_from_checkpoint(ckpt_path).cuda()
            print("Model loaded successfully on GPU")
        else:
            model_wrapper = SceneScriptWrapper.load_from_checkpoint(ckpt_path)
            print("Model loaded successfully on CPU")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model_wrapper = None


@router.get("/")
def read_root():
    return {"message": "SceneScript API is running", "status": "healthy"}


@router.get("/model/status", response_model=ModelStatus)
def get_model_status():
    """Get the current model loading status"""
    global model_wrapper
    return ModelStatus(
        loaded=model_wrapper is not None and SCENESCRIPT_AVAILABLE,
        model_path="./weights/scenescript_model_ase.ckpt" if model_wrapper else None,
        device="cuda" if model_wrapper and torch.cuda.is_available() else "cpu",
    )


@app.post("/inference", response_model=InferenceResponse)
def run_inference(data: PointCloudData):
    """Run SceneScript inference on point cloud data"""
    global model_wrapper

    if not SCENESCRIPT_AVAILABLE:
        raise HTTPException(status_code=503, detail="SceneScript modules not available.")

    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check model status.")

    try:
        import time

        start_time = time.time()

        # Convert input points to numpy array
        points = np.array(data.points, dtype=np.float32)

        if points.shape[1] != 3:
            raise HTTPException(status_code=400, detail="Points must have 3 coordinates [x, y, z]")

        # Run inference
        lang_seq = model_wrapper.run_inference(
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


@app.post("/inference/file")
def run_inference_from_uploaded_file(
    filename: str = Form(...),
    nucleus_sampling_thresh: float = Form(0.05),
    verbose: bool = Form(True),
):
    """Run SceneScript inference on previously uploaded point cloud file"""
    global model_wrapper

    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check model status.")

    try:
        import time

        start_time = time.time()

        # Construct the file path in the pointclouds directory
        pointclouds_dir = "/home/playboigeorgy/scenescript_clone_v3/pointclouds"
        file_path = os.path.join(pointclouds_dir, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Point cloud file not found: {filename}")

        # Load point cloud using the SceneScript PointCloud class
        point_cloud_obj = PointCloud.load_from_file(file_path)
        points = point_cloud_obj.points

        # Convert numpy array to torch tensor as expected by run_inference
        import torch

        if not torch.is_tensor(points):
            points = torch.FloatTensor(points)

        # Move to device if CUDA is available
        if torch.cuda.is_available() and hasattr(model_wrapper, "device"):
            points = points.to(model_wrapper.device)

        # Run inference using the model's run_inference method
        lang_seq = model_wrapper.run_inference(
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


@app.post("/inference/scene")
def run_inference_on_scene_file(
    file: UploadFile = File(...),  # File from FormData
    nucleus_sampling_thresh: float = Form(0.05),  # FormData field
    verbose: bool = Form(True),  # FormData field
):
    """Run SceneScript inference on uploaded scene file"""
    global model_wrapper

    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        import time

        start_time = time.time()
        # Save uploaded file temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and process the point cloud
        point_cloud_obj = PointCloud.load_from_file(temp_file_path)
        raw_point_cloud = torch.tensor(point_cloud_obj.points, dtype=torch.float32)

        # Run inference
        lang_seq = model_wrapper.run_inference(
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


@app.get("/pointclouds")
def list_available_pointclouds():
    """List available point cloud files"""
    pointclouds_dir = "/home/playboigeorgy/scenescript_clone_v3/pointclouds"
    try:
        files = [f for f in os.listdir(pointclouds_dir) if f.endswith((".csv", ".csv.gz"))]
        return {"available_files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.post("/upload")
async def upload_files(
    semidense_points: UploadFile = File(None), trajectory: UploadFile = File(None)
):
    """Upload and store point cloud and trajectory files"""
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = "/home/playboigeorgy/scenescript_clone_v3/uploaded_data"
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
            pointclouds_dir = "/home/playboigeorgy/scenescript_clone_v3/pointclouds"
            os.makedirs(pointclouds_dir, exist_ok=True)

            inference_filename = f"uploaded_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv.gz"
            inference_path = os.path.join(pointclouds_dir, inference_filename)

            # Copy the file for inference
            with open(inference_path, "wb") as buffer:
                with open(uploaded_files["semidense_points"]["path"], "rb") as source:
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
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/uploads")
def list_uploaded_files():
    """List all uploaded files"""
    uploads_dir = "/home/playboigeorgy/scenescript_clone_v3/uploaded_data"
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
        raise HTTPException(status_code=500, detail=f"Error listing uploaded files: {str(e)}")


@app.delete("/uploads/{filename}")
def delete_uploaded_file(filename: str):
    """Delete an uploaded file"""
    uploads_dir = "/home/playboigeorgy/scenescript_clone_v3/uploaded_data"
    file_path = os.path.join(uploads_dir, filename)

    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)
        return {"success": True, "message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
