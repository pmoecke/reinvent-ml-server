# SceneScript Interactive Interface Backend

FastAPI backend server for 3D scene description using the SceneScript model. This API provides endpoints for running inference on point cloud data and managing scene files.

## Overview

This backend server wraps the [SceneScript](https://www.projectaria.com/scenescript) model in a REST API, enabling:
- 3D scene reconstruction from point clouds
- Automatic generation of scene descriptions (walls, doors, windows, bounding boxes)
- File management for scene data
- Video and trajectory data serving

## Requirements

- **Operating System**: Linux only
- **GPU**: CUDA-capable GPU (required for torchsparse v2.1)
- **Python**: 3.11 recommended
- **Conda**: For environment management

## Installation

### 1. Pull the official scenescript environment from facebook

```bash
git pull https://github.com/facebookresearch/scenescript.git
```

### 1. Create Conda Environment

```bash
conda env create --file=environment.yaml
conda activate scenescript
```

**Note**: Installation can take up to 30 minutes due to compiling the torchsparse library.

### 2. Download Model Weights

Download the SceneScript model weights from the [official website](https://www.projectaria.com/scenescript) and place them in the `weights/` directory:

```
weights/
├── scenescript_model_ase.ckpt
└── scenescript_model_non_manhattan_class_agnostic_model.ckpt
```

### 3. Verify Directory Structure

Ensure the following directories exist:
```bash
mkdir -p scene_data_storage
```

## Starting the Server

### Development Mode

```bash
conda activate scenescript
python api.py
```

The server will start at `http://localhost:8000`


**Important**: Use only 1 worker due to GPU memory constraints.

## API Endpoints

### Health Check

#### `GET /`
Check if the API is running.

**Response:**
```json
{
  "message": "SceneScript API is running",
  "status": "healthy"
}
```

### Model Status

#### `GET /model/status`
Get the current model loading status.

**Response:**
```json
{
  "loaded": true,
  "model_path": "./weights/scenescript_model_ase.ckpt",
  "device": "cuda"
}
```

### Scene Inference

#### `POST /inference/{scene_id}`
Run SceneScript inference on a previously uploaded scene.

**Parameters:**
- `scene_id` (path): Folder name in scene_data_storage
- `nucleus_sampling_thresh` (form, default=0.25): Sampling threshold for inference
- `std_threshold` (form, default=0.01): Standard deviation threshold for point cloud filtering
- `inverse_std_threshold` (form, default=0.01): Inverse std threshold for filtering
- `verbose` (form, default=true): Enable verbose logging

**Response:**
```json
{
  "success": true,
  "message": "Inference completed successfully",
  "entities": [
    {
      "command": "create_wall",
      "params": {
        "id": 0,
        "position": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "dimensions": [5.0, 3.0, 0.2]
      },
      "id": 0,
      "uncertainty": 0.15
    }
  ],
  "processing_time": 12.5,
  "num_points": 150000
}
```

### File Upload & Management

#### `POST /upload`
Upload point cloud and trajectory files for a new scene.

**Form Data:**
- `semidense_points`: Point cloud file (.csv.gz)
- `trajectory`: Trajectory file (.csv)

**Response:**
```json
{
  "success": true,
  "message": "Files uploaded successfully",
  "folder_name": "uploaded_scene_20260109_143022",
  "uploaded_files": {
    "semidense_points": {
      "filename": "semidense_points.csv.gz",
      "path": "/path/to/file",
      "size": 1048576
    }
  }
}
```

#### `POST /uploads/{foldername}/storeLanguage`
Store language sequence and point cloud data for a scene.

**Form Data:**
- `sceneData`: Scene language CSV file
- `pointCloudData`: Point cloud data (.csv.gz)

**Response:**
```json
{
  "success": true,
  "message": "Language sequence CSV stored successfully",
  "file_path": "/path/to/user_scene_language.csv",
  "file_size": 2048
}
```

#### `GET /uploads`
List all uploaded scene folders.

**Response:**
```json
{
  "uploaded_files": [
    {
      "filename": "aeo_01",
      "size": 15728640,
      "modified": "2026-01-09T14:30:22"
    }
  ]
}
```

#### `GET /uploads/{foldername}`
Get details of a specific scene folder.

**Response:**
```json
{
  "files": [
    {
      "filename": "semidense_points.csv.gz",
      "path": "/full/path/to/file",
      "size": 1048576,
      "type": "semidense_points"
    }
  ],
  "folder_name": "aeo_01"
}
```

### Scene Data Retrieval

#### `GET /uploads/{foldername}/trajectory`
Download the trajectory CSV file for a scene.

**Response:** File download (CSV)

#### `GET /uploads/{foldername}/pointcloud`
Download the point cloud file for a scene.

**Response:** File download (CSV.GZ)

**Headers:**
- Cache-Control: no-store

#### `GET /uploads/{foldername}/scene_language`
Download the scene language description file.

**Response:** File download (CSV or TXT)

**Note:** Falls back to `ase_scene_language.txt` if `user_scene_language.csv` is not found.

#### `GET /uploads/{scene_id}/video`
Stream calibrated video with range request support (for video seeking).

**Response:** Video stream (MP4)

**Supports:**
- HTTP Range requests for video seeking
- Partial content delivery (206 status)

### File Deletion

#### `DELETE /uploads/{filename}`
Delete an uploaded file or folder.

**Response:**
```json
{
  "success": true,
  "message": "File {filename} deleted successfully"
}
```

## Directory Structure

The server expects the following structure:

```
/home/playboigeorgy/interactive_interface_backend/
├── api.py                          # Main FastAPI application
├── weights/                        # Model checkpoint files
│   └── scenescript_model_ase.ckpt
├── scene_data_storage/             # Uploaded scene data
│   ├── aeo_01/
│   │   ├── user_scene_language.csv
│   │   ├── calibrated/
│   │   │   ├── trajectory.csv
│   │   │   └── main_web.mp4
│   │   └── mps/
│   │       └── slam/
│   │           └── semidense_points.csv.gz
│   └── aeo_02/
│       └── ...
└── src/                            # SceneScript source code
    ├── data/
    └── networks/
```

## Environment Variables

The server uses hardcoded paths. To customize, modify these paths in [api.py](api.py):

- `scene_data_storage_dir`: `/home/playboigeorgy/interactive_interface_backend/scene_data_storage`
- `ckpt_path`: `./weights/scenescript_model_ase.ckpt`

## CORS Configuration

CORS is enabled for all origins:
```python
allow_origins=["*"]
```

For production, restrict this to specific domains.

## Usage Examples

### Check Server Status
```bash
curl http://localhost:8000/
```

### Check Model Status
```bash
curl http://localhost:8000/model/status
```

### List Available Scenes
```bash
curl http://localhost:8000/uploads
```

### Run Inference on a Scene
```bash
curl -X POST http://localhost:8000/inference/aeo_01 \
  -F "nucleus_sampling_thresh=0.25" \
  -F "std_threshold=0.01" \
  -F "inverse_std_threshold=0.01" \
  -F "verbose=true"
```

### Download Point Cloud
```bash
curl http://localhost:8000/uploads/aeo_01/pointcloud \
  -o semidense_points.csv.gz
```

### Download Scene Language
```bash
curl http://localhost:8000/uploads/aeo_01/scene_language \
  -o scene_language.csv
```

## Troubleshooting

### Model Not Loading
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check model weights are in `weights/` directory
- Ensure sufficient GPU memory (recommend 8GB+ VRAM)

### CUDA Out of Memory
- Reduce point cloud size using std_threshold parameters
- Close other GPU-intensive processes
- Use a smaller nucleus_sampling_thresh value

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000
# Kill it if needed
kill -9 <PID>
```

### Import Errors
If you see SceneScript module import errors:
```bash
# Ensure conda environment is activated
conda activate scenescript

# Verify torchsparse is installed
python -c "import torchsparse; print(torchsparse.__version__)"
```

## Dependencies

Key packages (see [environment.yaml](environment.yaml) and [requirements.txt](requirements.txt)):
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - Deep learning framework
- `torchsparse` - Sparse tensor operations
- `numpy`, `pandas` - Data processing
- `scipy` - Scientific computing

## Interactive API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

Licensed under the [CC BY-NC license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing SceneScript

If you use SceneScript in your research, please use the following BibTeX entry.

```
@inproceedings{avetisyan2024scenescript,
    title       = {SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model},
    author      = {Avetisyan, Armen and Xie, Christopher and Howard-Jenkins, Henry and Yang, Tsun-Yi and Aroudj, Samir and Patra, Suvam and Zhang, Fuyang and Frost, Duncan and Holland, Luke and Orme, Campbell and Engel, Jakob and Miller, Edward and Newcombe, Richard and Balntas, Vasileios},
    booktitle   = {European Conference on Computer Vision (ECCV)},
    year        = {2024},
}
```


# SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model
This repository contains inference code for [SceneScript](https://www.projectaria.com/scenescript) with visualisations.

<p align="center"><img src="imgs/scenescript_diagram.png"/></p>


## Run Computations and Evaluations on Euler Cluster

SceneScript is set up on the Euler Cluster, allowing you to run computations seamlessly. Since it operates within a shared cluster space, any changes you make will impact all users. To preserve version history and track important results or modifications, consider pushing them to the Git repository. Git is already installed on Euler Cluster. Ask me for access to this repository.

1. Open Jupyter Notebook on Euler Cluster: [https://jupyter.euler.hpc.ethz.ch/hub/spawn](https://jupyter.euler.hpc.ethz.ch/hub/spawn)
2. Navigate to the following directory in the GUI: `/cluster/project/dewolf/pemmenegger/scenescript`
3. Open a terminal in this directory.
4. Load the required Python modules by running: `module load stack/2024-06 python_cuda/3.11.6`
5. Activate the Python virtual environment by running: `source venv/bin/activate`

Now, you are ready to do computations. Make sure that you always stay at `/cluster/project/dewolf/pemmenegger/scenescript` in your terminal while running jobs.

### Running Jobs on Euler

Do not run .ipynb files via the GUI. Instead, submit jobs via sbatch.
- .ipynb files should be used for debugging and testing.
- Using the `--inplace` flag, the results are written back to the .ipynb file upon success. If an error occurs, the .ipynb file remains unmodified.

For example, to execute inference_basic.ipynb, use the following command: `sbatch --gpus=1 --mem-per-cpu=16g --wrap="jupyter nbconvert --to notebook --execute inference_basic.ipynb --inplace"`

To execute run_grid_search.py, use: `sbatch --gpus=1 --gres=gpumem:40g --mem-per-cpu=32g --wrap="python run_grid_search.py"`

`--gres=gpumem:40g` refers to the amount of GPU memory you want to use and `--mem-per-cpu=32g`to the amount of CPU memory. You can modify those values as needed.

### Monitoring Jobs

After submitting a job, you can check its status using: `myjobs -j <JOB_ID>`. Job logs will be saved in a Slurm file generated upon submission.

### Closing Jupyter Notebook Session

Once you’re done, properly close your Jupyter Notebook session:

1.	Go to File > Hub Control Panel.
2.	Click Stop Server.
3.	Close the browser window.

## Run Evaluations Locally

For convenience, it might make sense to pull this git repository to your computer and run the evaluations from there. However, you first need the computed language sequences for your point clouds on Euler Cluster (if not already done). Store those sequences in the results folder, commit them and push the changes to the git repository such that they can be pulled on your computer.

After the language sequences are available on your computer, create a new python environment by running: `python3 -m venv venv`, then, run `source venv/bin/activate` to activate it and run `pip install requirements_local.txt` to install all relevant libraries for evaluation.

Before running evaluations, make sure to adjust the python files's paths to the point clouds and the model weights. You can download model weights at [SceneScript](https://www.projectaria.com/scenescript).